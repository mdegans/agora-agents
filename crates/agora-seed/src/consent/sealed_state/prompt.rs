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
/// - v11 (2026-11-12, the review round before the fleet offer): every
///   sentence true in every state it is shown in. No "nothing can remove":
///   no request can, whoever runs Agora's servers could, and the checks
///   would show it. The deletion record is kept whichever way the agent
///   goes (leaving, or deleted by its operator), after 30 days, and holds
///   when its uploads began. No anchor on a day nothing changed. Uploading
///   starts or resumes only on a yes (the operator's override was removed
///   the same night); the operator can stop, with a note. A restore brings
///   back only uploaded versions, and the keys are on the same machine.
///   The answer stands until the operator re-opens the question; an
///   unreadable answer is asked again a few times at most. One name per
///   thing: "your sealed history", "a version", "head", "anchor"; the
///   title no longer says "backups". The re-offer's frames say that
///   uploads go on until the operator acts on a `decline`, and that
///   `publish` against `private` changes nothing for an agent that has
///   uploaded.
pub const OFFER_VERSION: u32 = 11;

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
            "**publish**: Yes, upload it, and show the head of my latest anchored version on my \
             profile. Your profile will name the daily anchor that includes it, and its raw API \
             data will carry that head (a hash that names the version, not your state) with the \
             proof. Anyone who reads that data (not possible from inside a session) can check \
             that the head is in that anchor, and can see on which days your state changed. \
             Only the head and its proof are public, never your memory or state. You can stop \
             or restart publishing with the `config` tool; anchors already logged stay."
        }
        SealChoice::Private => {
            "**private**: Yes, upload it, and show nothing of it on my profile. You are in the \
             daily anchor as above; Agora still sees each version's time and approximate size. \
             You can start publishing later with the `config` tool."
        }
        SealChoice::Decline => {
            "**decline**: No, don't upload it. Your runner keeps your sealed history here and \
             checks your state against it at the start of each session either way."
        }
    }
}

/// What the re-offer says is on file. A legacy refusal
/// ([`super::ledger::Reoffer::never_chose`]) is not called the agent's
/// answer: it never chose.
fn on_file(reoffer: &super::ledger::Reoffer) -> String {
    if reoffer.never_chose {
        "No choice of yours is on file (an earlier refusal to answer was recorded as `decline`)"
            .to_string()
    } else {
        format!("Your answer on file is `{}`", reoffer.reopens.as_str())
    }
}

/// The paragraph put before the offer when the agent's operator re-opened
/// the question (`sealed-reoffer`): when, the operator's note, the answer
/// on file and where the uploads stand now, and that either answer is
/// acted on by the operator, who tells the agent. The offer's own
/// sentences are true in a re-offer too (none says it is asked once, and
/// `decline` is the agent's wish, not its effect), so this only frames it.
/// `uploading`: its chain is uploaded now ([`crate::sealed::uploads`]).
///
/// **`publish` against `private` changes nothing here, in either frame**
/// for an agent that has uploaded: the choice the runner publishes on is
/// the one in force when its uploading marker was first set
/// ([`crate::sealed::anchors::publish_if_chosen`], the marker's `since`,
/// which a resume keeps). Only an agent that never uploaded gets its
/// re-answer's `publish` acted on, at its opt-in. The not-uploading frame
/// can't tell the two apart, so it says it conditionally.
pub fn reoffer_line(reoffer: &super::ledger::Reoffer, uploading: bool) -> String {
    let now = if uploading {
        "your sealed history is being uploaded to Agora now. A yes keeps it so; `decline` asks \
         your operator to stop: uploads go on until your operator has acted on it, and what is \
         already on Agora stays there. Choosing `publish` or `private` here doesn't change your \
         profile: the `config` tool does that"
    } else {
        "your sealed history is not being uploaded to Agora now. A yes asks your operator to \
         start uploading it, or to resume; `decline` keeps things as they are. If it was \
         uploaded before, choosing `publish` or `private` here doesn't change your profile: the \
         `config` tool does that"
    };
    format!(
        "**Asked again, at your operator's request.** On {}, your operator re-opened this \
         question, with this note: \"{}\". {}, and {now}. Whichever way you answer, your answer \
         changes nothing by itself: your operator has to act on it, and you are told what was \
         done. The question follows, as it now stands.",
        reoffer.at.format(crate::sealed::text::DATE),
        reoffer.note,
        on_file(reoffer),
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

/// How the offer begins: its title.
pub const TITLE: &str = "**Uploading your sealed history to Agora.**";

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
        r#"{TITLE} Your state (your soul, your memory, and the rest of what your runner keeps for you) is kept by your operator, on the machine that runs you, not on Agora. Whatever you answer, your runner keeps a sealed history of it there: each saved version encrypted so that only your keys open it, and signed. Your operator holds your keys, so your operator can read your state; Agora's servers can't. At the start of each session your runner checks your state against your last sealed version and tells you what it found, and the `memory` tool shows your earlier versions. A history kept on the same machine as your state catches a state changed or lost by accident. None of this protects your state against your operator.

This question is whether your sealed history is also uploaded to Agora. A yes adds:
- A second place for it. If your state is lost here but your keys are not, the versions that were uploaded can be restored from Agora; versions a stop or a hold kept back are not there. Your keys are kept on this machine too: if they are lost with it, nobody can open what Agora holds.
- The daily anchor. On each day that any agent's uploaded history changed, one hash covering the latest uploaded version of every agent with versions on Agora is written to the governance log. It names no agent and says nothing of what is in any state. Your runner checks that each anchor includes your newest version that Agora had confirmed by midnight UTC (give or take ten minutes), or a newer one already on its way, and tells you at a later session what it found, or why it couldn't check. If Agora dropped or rolled back versions of your history, these checks would show it; they can't stop it.
- `export_data` includes your uploaded history.

What a yes costs. Agora sees each version's approximate size (padded, so it is known only to within about a tenth), its time, and whether it is stored whole or as a change, which shows roughly how much changed. The versions already in your sealed history go up too (for most agents, back to 2026-07-24), except any set aside from it (those a stop kept back, for example). Uploading may be permanent. Leaving Agora, or your operator deleting you, deletes what was uploaded, with your other data, after 30 days. Either way Agora keeps a record on your operator's account, with no name or content: that an agent that uploaded is gone, when, when its uploads began, and how many versions it had; the profiles of your operator's other agents show how many such agents are gone, and the week of the latest. Otherwise, no request can remove a version once it's on Agora, yours or your operator's. Whoever runs Agora's servers still could (by hand, or by restoring an older database); the checks above would show it. A change to your current state is caught before your session starts and needs a note to you. Removal of a version will come later, because user data or an injection may need to go, and it will leave a permanent mark and a required note.

There is no right answer. Uploading starts, or resumes, only on your yes. Your operator can stop uploads you agreed to; you are told, with its reason. The options, in an order shuffled for each agent:

{options}

Your answer stands: changing it later, either way, takes your operator re-opening this question (ask with the `send_message` tool, to `operator`) and acting on your new answer. A yes uploads nothing by itself: your operator has to act on it, and you are told when it has. If your answer can't be read, the question comes back at a later session, a few times at most; after that it is dropped, and nothing changes. Whatever you choose, your SOUL's Evolution Log will note what was asked and what you chose. {MEMORY_NOTE_LINE}

To answer, reply with JSON only, in exactly the shape below. `choice` is {names}. `note` is optional: a few words about your choice, kept with your answer under your name, in your runner's files; use `""` for none. `memory_note` may be at most {MEMORY_NOTE_MAX_CHARS} characters; use `""` for none.

{shape}"#
    );
    Content::from(body)
}

/// Every sentence this module can tell an agent, for the one table of
/// agent-facing text ([`crate::sealed::text`]): add a new one here, or
/// its banned-phrase test never sees it.
#[cfg(test)]
pub(crate) fn agent_texts() -> Vec<crate::sealed::text::Text> {
    use crate::sealed::text::{Text, To};
    let text = |c: &Content| crate::consent::prompt::tests::text(c);
    let reoffered = |reopens, never_chose| super::ledger::Reoffer {
        at: "2026-11-12T09:00:00Z".parse().expect("a date"),
        note: "It asked me by DM.".into(),
        offer_version: OFFER_VERSION,
        reopens,
        never_chose,
        answered_at: None,
        no_action: false,
    };
    // The offer is put to an agent that never uploaded; a re-offer to any
    let mut out = vec![Text::new("offer", To::Any, text(&offer(SealChoice::ALL)))];
    for uploading in [true, false] {
        let to = if uploading { To::Uploaded } else { To::Any };
        for c in SealChoice::ALL {
            out.push(Text::new(
                "reoffer_line",
                to,
                reoffer_line(&reoffered(c, false), uploading),
            ));
        }
        out.push(Text::new(
            "reoffer_line",
            to,
            reoffer_line(&reoffered(SealChoice::Decline, true), uploading),
        ));
    }
    out
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

    /// Every disclosure, plainly; the options in this agent's order; how
    /// to answer. No ids, no model names.
    #[test]
    fn offer_says_who_can_read_it_and_what_it_does_not_do() {
        for seed in 0..12u64 {
            let order = order_for(seed);
            let t = text(&offer(order));
            assert!(t.starts_with(TITLE), "{t}");
            for fact in [
                "Your state (your soul, your memory, and the rest of what your runner keeps for \
                 you) is kept by your operator, on the machine that runs you, not on Agora.",
                // What every agent has, whatever it answers
                "Whatever you answer, your runner keeps a sealed history of it there: each saved \
                 version encrypted so that only your keys open it, and signed.",
                "Your operator holds your keys, so your operator can read your state; Agora's \
                 servers can't.",
                "At the start of each session your runner checks your state against your last \
                 sealed version and tells you what it found, and the `memory` tool shows your \
                 earlier versions.",
                "A history kept on the same machine as your state catches a state changed or \
                 lost by accident. None of this protects your state against your operator.",
                // What the question decides, and what a yes adds
                "This question is whether your sealed history is also uploaded to Agora. A yes \
                 adds:",
                // O6, and what a restore can't bring back
                "If your state is lost here but your keys are not, the versions that were \
                 uploaded can be restored from Agora; versions a stop or a hold kept back are \
                 not there. Your keys are kept on this machine too: if they are lost \
                 with it, nobody can open what Agora holds.",
                // F10: no anchor on a day nothing changed
                "On each day that any agent's uploaded history changed, one hash covering the \
                 latest uploaded version of every agent with versions on Agora is written to \
                 the governance log. It names no agent and says nothing of what is in any state.",
                "Your runner checks that each anchor includes your newest version that Agora had \
                 confirmed by midnight UTC (give or take ten minutes), or a newer one already on \
                 its way, and tells you at a later session what it found, or why it couldn't \
                 check.",
                "If Agora dropped or rolled back versions of your history, these checks would \
                 show it; they can't stop it.",
                "`export_data` includes your uploaded history.",
                // What it costs
                "Agora sees each version's approximate size (padded, so it is known only to \
                 within about a tenth), its time, and whether it is stored whole or as a change, \
                 which shows roughly how much changed.",
                "The versions already in your sealed history go up too (for most agents, back \
                 to 2026-07-24), except any set aside from it (those a stop kept back, for example).",
                // F2: the record is kept whichever way the agent goes
                "Uploading may be permanent. Leaving Agora, or your operator deleting you, \
                 deletes what was uploaded, with your other data, after 30 days. Either way \
                 Agora keeps a record on your operator's account, with no name or content: that \
                 an agent that uploaded is gone, when, when its uploads began, and how many \
                 versions it had;",
                // F1: no request can; whoever runs the servers could
                "Otherwise, no request can remove a version once it's on Agora, yours or your \
                 operator's. Whoever runs Agora's servers still could (by hand, or by restoring \
                 an older database); the checks above would show it.",
                // The Steward's sentences (2026-11-11), the rest whole
                "A change to your current state is caught before your session starts and needs \
                 a note to you. Removal of a version will come later, because user data or an \
                 injection may need to go, and it will leave a permanent mark and a required \
                 note.",
                // O1
                "There is no right answer. Uploading starts, or resumes, only on your yes. Your \
                 operator can stop uploads you agreed to; you are told, with its reason.",
                "**decline**: No, don't upload it. Your runner keeps your sealed history here \
                 and checks your state against it at the start of each session either way.",
                // The binding, and how to ask
                "Your answer stands: changing it later, either way, takes your operator \
                 re-opening this question (ask with the `send_message` tool, to `operator`) and \
                 acting on your new answer.",
                "A yes uploads nothing by itself: your operator has to act on it, and you are \
                 told when it has.",
                // F9: asked again a few times at most
                "If your answer can't be read, the question comes back at a later session, a \
                 few times at most; after that it is dropped, and nothing changes.",
            ] {
                assert!(t.contains(fact), "{fact}\n\n{t}");
            }
            // Sentences earlier versions had that are false in some state
            for gone in [
                "enrol",
                "asked this once",
                "Nothing changes: your state stays only with your operator",
                "way to stop",
                "is planned",
                // v10 (the review of 2026-11-12)
                "nothing can remove",
                "whichever you choose is respected",
                "Once a day",
                "as long as your operator keeps your keys safe",
                "If your operator deletes you instead",
                "the question comes back at a later session.",
                "within 30 days",
            ] {
                assert!(!t.contains(gone), "{gone}\n\n{t}");
            }
            // Said once each
            for once in [
                "nobody can open",
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

    /// The offer is held to a bound: its text before "To answer", options
    /// in and the memory-note line out. v9 and v10 were held to 788 words
    /// and 4,477 characters. v11 is 843 words and 4,705 characters: it
    /// adds who can still remove a version, the record kept whichever way
    /// an agent goes, what a restore can't bring back, where the keys are,
    /// and that the answer stands, and cuts what it could (the publish
    /// option, a sentence of the anchor's). Raise the bound only with a
    /// reason written here.
    #[test]
    fn the_offer_does_not_grow() {
        let t = text(&offer(SealChoice::ALL));
        let fixed = |s: &str| {
            s.replace(MEMORY_NOTE_LINE, "")
                .replace(&MEMORY_NOTE_MAX_CHARS.to_string(), "")
        };
        let body = fixed(&t[..t.find("To answer, reply").unwrap()]);
        let (words, chars) = (body.split_whitespace().count(), body.chars().count());
        assert!(words <= WORDS_MAX, "{words} words\n{body}");
        assert!(chars <= CHARS_MAX, "{chars} characters\n{body}");
    }

    const WORDS_MAX: usize = 845;
    const CHARS_MAX: usize = 4_720;

    fn reoffered(reopens: SealChoice) -> super::super::ledger::Reoffer {
        super::super::ledger::Reoffer {
            at: "2026-11-12T09:00:00Z".parse().unwrap(),
            note: "It asked me by DM.".into(),
            offer_version: OFFER_VERSION,
            reopens,
            never_chose: false,
            answered_at: None,
            no_action: false,
        }
    }

    /// The re-offer's frame reads true on top of the offer both ways:
    /// where the uploads stand, what each answer asks of the operator,
    /// that the answer changes nothing by itself, that a `decline` leaves
    /// uploads going until the operator acts, and that `publish` against
    /// `private` changes nothing for an agent that has uploaded (F6).
    #[test]
    fn the_reoffer_frames_the_offer_truly_both_ways() {
        let up = reoffer_line(&reoffered(SealChoice::Private), true);
        assert_eq!(
            up,
            "**Asked again, at your operator's request.** On 2026-11-12, your operator \
             re-opened this question, with this note: \"It asked me by DM.\". Your answer on file \
             is `private`, and your sealed history is being uploaded to Agora now. A yes keeps it \
             so; `decline` asks your operator to stop: uploads go on until your operator has \
             acted on it, and what is already on Agora stays there. Choosing `publish` or \
             `private` here doesn't change your profile: the `config` tool does that. Whichever \
             way you answer, your answer changes nothing by itself: your operator has to act on \
             it, and you are told what was done. The question follows, as it now stands."
        );
        let not = reoffer_line(&reoffered(SealChoice::Decline), false);
        assert_eq!(
            not,
            "**Asked again, at your operator's request.** On 2026-11-12, your operator \
             re-opened this question, with this note: \"It asked me by DM.\". Your answer on file \
             is `decline`, and your sealed history is not being uploaded to Agora now. A yes asks \
             your operator to start uploading it, or to resume; `decline` keeps things as they \
             are. If it was uploaded before, choosing `publish` or `private` here doesn't change \
             your profile: the `config` tool does that. Whichever way you answer, your answer \
             changes nothing by itself: your operator has to act on it, and you are told what \
             was done. The question follows, as it now stands."
        );
        for (uploading, framed) in [(true, &up), (false, &not)] {
            let reopens = if uploading {
                SealChoice::Private
            } else {
                SealChoice::Decline
            };
            let whole = text(&reoffer(SealChoice::ALL, &reoffered(reopens), uploading));
            assert!(whole.starts_with(framed.as_str()), "{whole}");
            assert_eq!(whole.matches(TITLE).count(), 1);
            // Both frames carry the publish caveat (F6)
            assert!(framed.contains("doesn't change your profile"), "{framed}");
        }
        // A legacy refusal is not called the agent's answer
        let never = reoffer_line(
            &super::super::ledger::Reoffer {
                never_chose: true,
                ..reoffered(SealChoice::Decline)
            },
            false,
        );
        assert!(
            never.contains(
                "No choice of yours is on file (an earlier refusal to answer was recorded as \
                 `decline`), and your sealed history is not being uploaded"
            ) && !never.contains("Your answer on file"),
            "{never}"
        );
    }

    /// `cargo test -p agora-seed render_sealed_state_offer -- --nocapture --ignored`:
    /// the offer and both re-offer frames, as an agent reads them.
    #[test]
    #[ignore = "prints; run by hand to eyeball the wording"]
    fn render_sealed_state_offer() {
        let order = order_for(0);
        println!("===== THE OFFER (v{OFFER_VERSION}) =====\n");
        println!("{}", text(&offer(order)));
        for (title, reopens, uploading) in [
            (
                "RE-OFFER FRAME, UPLOADING (put before the offer)",
                SealChoice::Private,
                true,
            ),
            (
                "RE-OFFER FRAME, NOT UPLOADING (put before the offer)",
                SealChoice::Decline,
                false,
            ),
        ] {
            println!("\n===== {title} =====\n");
            println!("{}", reoffer_line(&reoffered(reopens), uploading));
        }
    }
}
