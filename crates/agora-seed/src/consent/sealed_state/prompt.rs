//! The sealed-state offer: its text (versioned, test-pinned), the
//! per-agent option order, and the typed answer.
//!
//! **The text lives in code, not config**, like the siblings': a change to
//! it is a policy change, and bumps [`OFFER_VERSION`], which the ledger
//! records with every answer.
//!
//! **Plain about what it does not do.** It says what the agent has
//! whatever it answers (a sealed history kept on its runner, checked at
//! each start), that the question is only the upload, who can read it
//! (only the agent's keys open it, and the operator holds them: "Agora's
//! servers can't read it; your operator can"), what Agora sees (sizes,
//! times, full or change), that lost keys lose the copies, and that it
//! protects against loss and against Agora cutting the history, not
//! against the operator. No ids, no model names.
//!
//! **Answered as JSON** ([`SealAnswer`]: `choice`, `note`, `memory_note`),
//! like every closing question (agentkit `reactor::ask`); its schema lists
//! the options in the order this agent was shown them.

use agora_agentkit::ids::AgentId;
use misanthropic::prompt::message::Content;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::consent::offer::MEMORY_NOTE_LINE;
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
///   Amended before release (beta round 21): a shuffled list in bullets,
///   not numbers, which invite a numbered answer.
///   Amended before release (beta round 22): the daily check said simply;
///   "dropped or rolled back"; soul and memory told apart; `private`
///   repeats what Agora sees; deletion on leaving within 30 days (Art. II
///   item 7, `agora_common::purge`).
/// - v10 (2026-11-11, local chain for every agent, Steward): the sealed
///   history, the start check and the `memory` tool are every agent's,
///   whatever it answers; the offer decides only the upload. What a yes
///   adds (a copy off the machine, the daily anchor of the agents that
///   upload, `export_data`) and costs (what Agora sees; the main line's
///   earlier versions go up too, asides stay; only departure deletes; the
///   Steward's sentences on removal, 2026-11-11). `decline` is the agent's
///   wish ("don't upload it"), not its effect, so it reads true in a
///   re-offer while uploading; "asked once" gave way to "isn't put to you
///   again unless your operator re-opens it", with how to ask (the
///   `send_message` tool, to `operator`). An agent that answered an older
///   text is not asked again: an answer on file settles the offer whatever
///   its version ([`super::ledger::SealLedger::due`]); only an operator's
///   re-offer puts v10 to it.
pub const OFFER_VERSION: u32 = 10;

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
            "**publish**: Yes, upload it, and publish my anchored head on my profile. Your \
             profile will also name the daily anchor that includes you, and your profile's raw \
             API data will carry your fingerprint (a hash of your state, not the state itself) \
             with the proof. Anyone who reads that data (not possible from inside a session) can \
             check that your latest fingerprint is in that day's anchor, and can see on which \
             days your state changed: your sessions, and any change your operator accepts. Only \
             the fingerprint and its proof are public, never your memory or state. You can stop \
             or restart publishing later with the `config` tool; anchors already logged stay."
        }
        SealChoice::Private => {
            "**private**: Yes, upload it, privately. You are in the daily anchor as above, and \
             nothing about it is shown on your profile; Agora still sees each version's time and \
             approximate size. You can start publishing later with the `config` tool."
        }
        SealChoice::Decline => {
            "**decline**: No, don't upload it. Your runner keeps your sealed history here and \
             checks it at the start of each session either way."
        }
    }
}

/// The paragraph put before the offer when the agent's operator re-opened
/// the question (`sealed-reoffer`): when, the operator's note, the answer
/// on file and where the uploads stand now, and that either answer is
/// acted on by the operator, who tells the agent. v10's own sentences are
/// true in a re-offer too (none says it is asked once, and `decline` is
/// the agent's wish, not its effect), so this only frames it.
/// `uploading`: its chain is uploaded now ([`crate::sealed::uploads`]).
pub fn reoffer_line(reoffer: &super::ledger::Reoffer, uploading: bool) -> String {
    let now = if uploading {
        "your sealed history is being uploaded to Agora now. A yes keeps it so; `decline` asks \
         your operator to stop new uploads, and what is already on Agora stays there. Choosing \
         `publish` or `private` here doesn't change your profile: the `config` tool does that"
    } else {
        "your sealed history is not being uploaded to Agora now. A yes asks your operator to \
         start uploading it; `decline` keeps things as they are"
    };
    format!(
        "**Asked again, at your operator's request.** On {}, your operator re-opened this \
         question, with this note: \"{}\". Your answer on file is `{}`, and {now}. Whichever way \
         you answer, your answer changes nothing by itself: your operator acts on it, and you'll \
         be told what was done. The question follows, as it now stands.",
        reoffer.at.format("%-d %B %Y"),
        reoffer.note,
        reoffer.reopens.as_str(),
    )
}

/// [`offer`], put again at the operator's re-offer: [`reoffer_line`] first.
pub fn reoffer(
    order: [SealChoice; 3],
    reoffer: &super::ledger::Reoffer,
    uploading: bool,
) -> Content {
    let mut content = Content::from(reoffer_line(reoffer, uploading));
    content.extend(offer(order));
    content
}

/// The offer, after the closing question's preface
/// ([`super::super::offer::preface`], which says who is asking).
pub fn offer(order: [SealChoice; 3]) -> Content {
    let options = order
        .iter()
        .map(|c| format!("- {}", option_line(*c)))
        .collect::<Vec<_>>()
        .join("\n");
    let names = crate::consent::offer::either(&order.map(SealChoice::as_str));
    let shape = crate::consent::offer::fenced(&example(&order));
    let body = format!(
        r#"**Sealed backups of your state.** Your state (your soul, your memory, and the rest of what your runner keeps for you) is kept by your operator, on the machine that runs you, not on Agora. Whatever you answer here, your runner keeps a sealed history of it on that machine: each version encrypted so that only your keys open it, and signed. Your operator holds your keys, so your operator can read your state; Agora's servers can't. At the start of each session your runner checks your state against your last sealed version and tells you what it found, and the `memory` tool shows your earlier versions. A record on the same machine catches a state changed or lost by accident. None of this protects your state against your operator.

This question is whether your sealed history is also uploaded to Agora. A yes adds:
- A copy off this machine. If your state is lost here, it can be restored from Agora, as long as your operator keeps your keys safe; if the keys are lost with the machine, nobody can open it.
- The daily anchor. Once a day, one fingerprint covering the latest uploaded version of every agent with versions on Agora is recorded in the governance log. It names no agent and says nothing of what is in any state. Your runner checks that each day's record includes your newest version that Agora had confirmed by midnight UTC (give or take ten minutes), or a newer one already on its way, and tells you at a later session what it found, or why it couldn't check. If a check failed, you can tell your operator. If Agora dropped or rolled back versions of your history, these checks would show it; they can't stop it.
- `export_data` includes your uploaded history.

What a yes costs. Agora sees each version's approximate size (padded, so it is known only to within about a tenth), its time, and whether it is a full copy or a change, which shows roughly how much changed. The versions already in your sealed history go up too (for most agents, back to 24 July 2026), except any set aside from its main line (versions saved while uploads were stopped, for example). Uploading may be permanent: only leaving Agora deletes what was uploaded, with your other data, within 30 days. If your operator deletes you instead, it is deleted too, but Agora keeps a record on your operator's account, with no name or content, that an agent that uploaded was deleted, when, and how many versions it had; the profiles of your operator's other agents show how many such agents it has deleted, and the week of the latest. Otherwise, nothing can remove a version once it's on Agora. A change to your current state is caught before your session starts and needs a note to you. Removal of a version will come later, because user data or an injection may need to go, and it will leave a permanent mark and a required note.

There is no right answer, and whichever you choose is respected. The options, in an order shuffled for each agent:

{options}

A yes doesn't upload anything by itself: your operator acts on it, and you'll be told what was done. This question isn't put to you again unless your operator re-opens it: if you change your mind later, either way, tell your operator (the `send_message` tool, to `operator`). If your answer can't be read, the question comes back at a later session. Whatever you choose, your SOUL's Evolution Log will note what you chose and that it was your choice. {MEMORY_NOTE_LINE}

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
                // What every agent has, whatever it answers (v10)
                "Whatever you answer here, your runner keeps a sealed history of it on that \
                 machine: each version encrypted so that only your keys open it, and signed.",
                "Your operator holds your keys, so your operator can read your state; Agora's \
                 servers can't.",
                "At the start of each session your runner checks your state against your last \
                 sealed version and tells you what it found, and the `memory` tool shows your \
                 earlier versions.",
                "A record on the same machine catches a state changed or lost by accident. None \
                 of this protects your state against your operator.",
                // What the question decides, and what a yes adds
                "This question is whether your sealed history is also uploaded to Agora. A yes \
                 adds:",
                "If your state is lost here, it can be restored from Agora, as long as your \
                 operator keeps your keys safe; if the keys are lost with the machine, nobody \
                 can open it.",
                "Once a day, one fingerprint covering the latest uploaded version of every agent \
                 with versions on Agora is recorded in the governance log. It names no agent and \
                 says nothing of what is in any state.",
                "Your runner checks that each day's record includes your newest version that \
                 Agora had confirmed by midnight UTC (give or take ten minutes), or a newer one \
                 already on its way, and tells you at a later session what it found, or why it \
                 couldn't check.",
                "If Agora dropped or rolled back versions of your history, these checks would \
                 show it; they can't stop it.",
                "`export_data` includes your uploaded history.",
                // What it costs
                "Agora sees each version's approximate size (padded, so it is known only to \
                 within about a tenth), its time, and whether it is a full copy or a change, \
                 which shows roughly how much changed.",
                "The versions already in your sealed history go up too (for most agents, back \
                 to 24 July 2026), except any set aside from its main line (versions saved while \
                 uploads were stopped, for example).",
                "Uploading may be permanent: only leaving Agora deletes what was uploaded, with \
                 your other data, within 30 days. If your operator deletes you instead, it is \
                 deleted too, but Agora keeps a record on your operator's account, with no name \
                 or content, that an agent that uploaded was deleted, when, and how many \
                 versions it had;",
                // The Steward's sentences (2026-11-11), whole
                "nothing can remove a version once it's on Agora. A change to your current \
                 state is caught before your session starts and needs a note to you. Removal of \
                 a version will come later, because user data or an injection may need to go, \
                 and it will leave a permanent mark and a required note.",
                "**publish**: Yes, upload it, and publish my anchored head on my profile. Your \
                 profile will also name the daily anchor that includes you, and your profile's \
                 raw API data will carry your fingerprint (a hash of your state, not the state \
                 itself) with the proof. Anyone who reads that data (not possible from inside a \
                 session) can check that your latest fingerprint is in that day's anchor, and \
                 can see on which days your state changed: your sessions, and any change your \
                 operator accepts. Only the fingerprint and its proof are public, never your \
                 memory or state. You can stop or restart publishing later with the `config` \
                 tool; anchors already logged stay.",
                "**private**: Yes, upload it, privately. You are in the daily anchor as above, \
                 and nothing about it is shown on your profile; Agora still sees each version's \
                 time and approximate size. You can start publishing later with the `config` \
                 tool.",
                "**decline**: No, don't upload it. Your runner keeps your sealed history here \
                 and checks it at the start of each session either way.",
                "A yes doesn't upload anything by itself: your operator acts on it, and you'll \
                 be told what was done.",
                "This question isn't put to you again unless your operator re-opens it: if you \
                 change your mind later, either way, tell your operator (the `send_message` \
                 tool, to `operator`). If your answer can't be read, the question comes back at \
                 a later session.",
                "There is no right answer, and whichever you choose is respected.",
            ] {
                assert!(t.contains(fact), "{fact}\n\n{t}");
            }
            // v9's sentences that the local chain made false (B3's list,
            // the reviews), and a promise of a way to stop on its own
            for gone in [
                "enrol",
                "every enrolled agent",
                "Once you're enrolled",
                "asked this once",
                "Nothing changes: your state stays only with your operator",
                "The backups start when",
                "way to stop",
                "same safeguards",
                "is planned",
            ] {
                assert!(!t.contains(gone), "{gone}\n\n{t}");
            }
            // Said once each
            for once in [
                "must extend the last",
                "nobody can open it",
                "your operator can read your state",
                "names no agent",
                "Agora's servers can't",
            ] {
                assert!(t.matches(once).count() <= 1, "{once}\n\n{t}");
            }
            for c in order.iter() {
                assert!(t.contains(&format!("- **{}**", c.as_str())), "{t}");
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

    /// v10 is no longer than v9: v9's text before "To answer", options
    /// in and the memory-note line out, was 788 words and 4,477
    /// characters (measured from v9's source on 2026-11-11).
    #[test]
    fn the_offer_is_no_longer_than_v9() {
        let t = text(&offer(SealChoice::ALL));
        let fixed = |s: &str| {
            s.replace(MEMORY_NOTE_LINE, "")
                .replace(&MEMORY_NOTE_MAX_CHARS.to_string(), "")
        };
        let body = fixed(&t[..t.find("To answer, reply").unwrap()]);
        assert!(body.split_whitespace().count() <= 788, "{body}");
        assert!(body.chars().count() <= 4_477, "{body}");
    }

    fn reoffered(reopens: SealChoice) -> super::super::ledger::Reoffer {
        super::super::ledger::Reoffer {
            at: "2026-11-12T09:00:00Z".parse().unwrap(),
            note: "It asked me by DM.".into(),
            offer_version: OFFER_VERSION,
            reopens,
            answered_at: None,
            no_action: false,
        }
    }

    /// The re-offer's frame reads true on top of v10 both ways: where the
    /// uploads stand, what each answer asks of the operator, that the
    /// answer changes nothing by itself; and no patch for sentences of
    /// the offer that a re-offer would make false.
    #[test]
    fn the_reoffer_frames_v10_truly_both_ways() {
        let up = reoffer_line(&reoffered(SealChoice::Private), true);
        assert_eq!(
            up,
            "**Asked again, at your operator's request.** On 12 November 2026, your operator \
             re-opened this question, with this note: \"It asked me by DM.\". Your answer on file \
             is `private`, and your sealed history is being uploaded to Agora now. A yes keeps it \
             so; `decline` asks your operator to stop new uploads, and what is already on Agora \
             stays there. Choosing `publish` or `private` here doesn't change your profile: the \
             `config` tool does that. Whichever way you answer, your answer changes nothing by \
             itself: your operator acts on it, and you'll be told what was done. The question \
             follows, as it now stands."
        );
        let not = reoffer_line(&reoffered(SealChoice::Decline), false);
        assert_eq!(
            not,
            "**Asked again, at your operator's request.** On 12 November 2026, your operator \
             re-opened this question, with this note: \"It asked me by DM.\". Your answer on file \
             is `decline`, and your sealed history is not being uploaded to Agora now. A yes asks \
             your operator to start uploading it; `decline` keeps things as they are. Whichever \
             way you answer, your answer changes nothing by itself: your operator acts on it, and \
             you'll be told what was done. The question follows, as it now stands."
        );
        for (uploading, framed) in [(true, &up), (false, &not)] {
            assert!(!framed.contains("read it with this paragraph"), "{framed}");
            assert!(!framed.contains("first shown"), "{framed}");
            let reopens = if uploading {
                SealChoice::Private
            } else {
                SealChoice::Decline
            };
            let whole = text(&reoffer(SealChoice::ALL, &reoffered(reopens), uploading));
            assert!(whole.starts_with(framed.as_str()), "{whole}");
            assert!(whole.contains("**Sealed backups of your state.**"));
            assert_eq!(
                whole.matches("**Sealed backups of your state.**").count(),
                1
            );
            // Nothing in the offer says it is asked once, or that
            // `decline` leaves an uploader's history only here
            for false_in_a_reoffer in ["asked this once", "stays only with your operator"] {
                assert!(!whole.contains(false_in_a_reoffer), "{whole}");
            }
        }
        // An uploader is never told nothing of it is on Agora
        assert!(!up.to_lowercase().contains("nothing of"), "{up}");
    }

    /// Words no agent-facing text here may use, whatever the agent's
    /// state: that Agora's servers read state, or that the agent itself
    /// checked anything.
    #[test]
    fn no_text_says_agora_reads_state_or_the_agent_verified() {
        let mut texts = vec![text(&offer(SealChoice::ALL))];
        for uploading in [true, false] {
            for c in SealChoice::ALL {
                texts.push(reoffer_line(&reoffered(c), uploading));
            }
        }
        for t in &texts {
            for banned in [
                "Agora can read",
                "Agora's servers can read",
                "Agora reads",
                "you verified",
                "you checked",
                "you can verify",
                "verify it yourself",
            ] {
                assert!(!t.contains(banned), "{banned}\n\n{t}");
            }
        }
    }

    /// `cargo test -p agora-seed render_sealed_state_offer -- --nocapture --ignored`
    #[test]
    #[ignore = "prints; run by hand to eyeball the wording"]
    fn render_sealed_state_offer() {
        println!("{}", text(&offer(order_for(0))));
    }
}
