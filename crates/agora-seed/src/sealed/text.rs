//! The words agents are told about their sealed history: one name per
//! thing, one date format, and (in tests) one table of every agent-facing
//! sentence with one list of what none may say.
//!
//! **Vocabulary.** Each thing has one name, in every text an agent reads:
//!
//! | Thing | Name |
//! |---|---|
//! | the agent's soul, memory and the rest, as it is now | "your state" |
//! | the signed chain of sealed saves | "your sealed history" |
//! | one entry of it | "a version" ("a sealed version" where no "sealed history" stands near) |
//! | a version's short hash | its "head" |
//! | the latest version | "your last sealed version" |
//! | the operator's machine | "here" ("here, on your operator's machine" where a text may be read alone) |
//! | who runs the runner | "your operator" |
//! | the program, as the actor | "your runner" |
//! | what Agora holds of the history | "Agora's copy" (the only "copy") |
//! | the daily entry in the governance log | "the daily anchor", "an anchor" |
//! | versions off the line that goes on | "set aside" (said with why) |
//! | an agent whose history never went to Agora | "your runner does not upload it, and its files show no version of it ever uploaded to Agora" |
//!
//! "Record" is kept for one thing, what Agora keeps of a departed agent
//! on its operator's account (the offer). Dates are [`DATE`].

/// The one date format in agent-facing text: `2026-11-12`, as the verdict
/// and the `memory` tool (agentkit's renderer) write dates.
pub const DATE: &str = "%Y-%m-%d";

/// [`DATE`] with the time, where a time is said.
pub const DATE_TIME: &str = "%Y-%m-%d at %H:%M UTC";

/// How an agent reaches its operator: said where the runner can't know
/// that its operator was told, instead of claiming it.
pub const TELL_OPERATOR: &str =
    "You can tell your operator (the `send_message` tool, to `operator`).";

/// The kind of agent a text can be shown to: what it may say of Agora
/// depends on it.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum To {
    /// One with no uploading marker: nothing is asked of Agora for it, so
    /// nothing may be said of anchors, of being in step, or of uploads
    /// waiting.
    Never,
    /// One that uploads now, or did (stopped, held, or its marker in
    /// doubt): never told that nothing of it was uploaded.
    Uploaded,
    /// Either.
    Any,
}

/// One agent-facing text, as an agent would read it.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct Text {
    /// The constant or function that makes it.
    pub source: &'static str,
    pub to: To,
    pub text: String,
}

#[cfg(test)]
impl Text {
    pub(crate) fn new(source: &'static str, to: To, text: impl Into<String>) -> Self {
        Self {
            source,
            to,
            text: text.into(),
        }
    }
}

/// What no agent-facing text may say, whatever the agent's state, each
/// with why. Matched in lower case. **The one list**: a module's own test
/// pins its sentences and checks what is particular to them; what no
/// sentence may say is here.
#[cfg(test)]
pub(crate) const BANNED: &[(&str, &str)] = &[
    // Overclaims
    ("nothing is lost", "versions may be on one side only"),
    ("nothing of yours", "an agent's posts are on Agora"),
    (
        "nothing of your state is on agora",
        "decided without asking Agora",
    ),
    ("nothing of it is on agora", "decided without asking Agora"),
    ("nothing can remove", "whoever runs Agora's servers can"),
    ("can't be removed", "whoever runs Agora's servers can"),
    ("cannot be removed", "whoever runs Agora's servers can"),
    ("guarantee", "nothing here is one"),
    ("tamper-proof", "the operator holds the disk and the keys"),
    ("tamperproof", "the operator holds the disk and the keys"),
    ("safe from your operator", "it is not"),
    (
        "is respected",
        "the operator can stop uploads against a yes",
    ),
    // An alert is claimed only as sent (`alerts::ALERT_SENT`)
    (
        "has been told",
        "said of an alert that may not have gone out",
    ),
    (
        "have been told",
        "said of an alert that may not have gone out",
    ),
    (
        "has alerted your operator",
        "said of an alert that may not have gone out",
    ),
    // Who reads and who checks
    ("agora can read", "Agora's servers can't read state"),
    (
        "agora's servers can read",
        "Agora's servers can't read state",
    ),
    ("agora reads", "Agora's servers can't read state"),
    ("you verified", "the runner checks, not the agent"),
    ("you checked", "the runner checks, not the agent"),
    ("you can verify", "not from inside a session"),
    ("verify it yourself", "not from inside a session"),
    // One name per thing
    ("backup", "it is \"your sealed history\""),
    ("backed up", "it is \"your sealed history\""),
    (
        "your sealed versions",
        "the whole is \"your sealed history\"",
    ),
    (
        "uploading your sealed state",
        "what is uploaded is \"your sealed history\"",
    ),
    ("whoever runs your runner", "it is \"your operator\""),
    ("on this runner", "the place is \"here\""),
    ("here on your runner", "the place is \"here\""),
    ("here, on your runner", "the place is \"here\""),
    (
        "local cop",
        "\"copy\" is Agora's; here they are \"sealed files\"",
    ),
    (
        "sealed cop",
        "\"copy\" is Agora's; here they are \"sealed files\"",
    ),
    (
        "enrol",
        "agents are not enrolled: every agent has a sealed history",
    ),
    ("blob", "the operator's word"),
    ("outbox", "the operator's word"),
    ("genesis", "the operator's word"),
    ("keyframe", "the operator's word"),
    // No model is named
    ("claude", "no model is named"),
    ("haiku", "no model is named"),
    ("sonnet", "no model is named"),
    ("qwen", "no model is named"),
];

/// What a text for an agent that never uploaded may not say besides.
#[cfg(test)]
pub(crate) const BANNED_NEVER: &[&str] =
    &["in step", "anchor", "anc-", "waiting to upload", "are held"];

/// What a text for an agent that uploads, or did, may not say besides.
#[cfg(test)]
pub(crate) const BANNED_UPLOADED: &[&str] = &[
    "no version of it ever uploaded",
    "none of its",
    "does not upload your sealed history",
];

/// Month names: a date written with one is not [`DATE`].
#[cfg(test)]
const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Why `text` may not be shown to `to`, if it may not: the first banned
/// phrase it holds, or a date not written as [`DATE`].
#[cfg(test)]
pub(crate) fn fault(to: To, text: &str) -> Option<String> {
    // The file and commit of the verifier name agentkit's repository
    let lower = text.replace(super::check::VERIFIER, "").to_lowercase();
    if let Some((phrase, why)) = BANNED.iter().find(|(p, _)| lower.contains(p)) {
        return Some(format!("{phrase:?} ({why})"));
    }
    let more: &[&str] = match to {
        To::Never => BANNED_NEVER,
        To::Uploaded => BANNED_UPLOADED,
        To::Any => &[],
    };
    if let Some(phrase) = more.iter().find(|p| lower.contains(*p)) {
        return Some(format!("{phrase:?} (not to an agent of kind {to:?})"));
    }
    // "12 November 2026": a day, a month's name, a year
    let words: Vec<&str> = lower.split_whitespace().collect();
    for w in words.windows(3) {
        let digits = |s: &str, n: usize| {
            let s = s.trim_matches(|c: char| !c.is_ascii_digit());
            (1..=n).contains(&s.len()) && s.chars().all(|c| c.is_ascii_digit())
        };
        if digits(w[0], 2) && MONTHS.contains(&w[1]) && digits(w[2], 4) {
            return Some(format!("a date not written as {DATE}: {:?}", w.join(" ")));
        }
    }
    None
}

/// Panics if `text` may not be shown to `to` ([`fault`]): for a module's
/// own test of a text built from a live check.
#[cfg(test)]
pub(crate) fn assert_may_say(to: To, text: &str) {
    if let Some(why) = fault(to, text) {
        panic!("{why} in: {text}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The one table**: every agent-facing text about the sealed
    /// history, from every module that makes one, with the kind of agent
    /// it is shown to.
    fn table() -> Vec<Text> {
        let mut all = Vec::new();
        all.extend(crate::consent::sealed_state::prompt::agent_texts());
        all.extend(crate::consent::sealed_state::agent_texts());
        all.extend(crate::consent::sealed_state::ledger::agent_texts());
        all.extend(crate::sealed::check::agent_texts());
        all.extend(
            crate::sealed::notice::tests::every_line()
                .into_iter()
                .map(|(to, line)| Text::new("Notice::line", to, line)),
        );
        all.extend(crate::sealed::notice::agent_texts());
        all.extend(crate::sealed::healed::agent_texts());
        all.extend(crate::sealed::memory_tool::agent_texts());
        all.extend(crate::sealed::version::agent_texts());
        all.extend(crate::sealed::lost::agent_texts());
        all.extend(crate::sealed::stop::agent_texts());
        all.extend(crate::sealed::sizes::agent_texts());
        all.push(Text::new(
            "alerts::told",
            To::Any,
            crate::alerts::told(true),
        ));
        all.push(Text::new(
            "alerts::told",
            To::Any,
            crate::alerts::told(false),
        ));
        all.push(Text::new("TELL_OPERATOR", To::Any, TELL_OPERATOR));
        all
    }

    /// `cargo test -p agora-seed render_agent_text_table -- --nocapture --ignored`:
    /// the table, one text a line (source, kind of agent, text).
    #[test]
    #[ignore = "prints; run by hand to read every agent-facing text"]
    fn render_agent_text_table() {
        for t in table() {
            println!(
                "@@\t{}\t{:?}\t{}",
                t.source,
                t.to,
                t.text.replace('\n', "\\n")
            );
        }
    }

    /// No text in the table says anything on the one banned list, or
    /// what its kind of agent may not be told, or a date in another form.
    #[test]
    fn no_agent_facing_text_says_what_it_may_not() {
        let table = table();
        assert!(table.len() > 200, "{} texts", table.len());
        let faults: Vec<String> = table
            .iter()
            .filter_map(|t| {
                fault(t.to, &t.text).map(|why| format!("{}: {why}\n    {}", t.source, t.text))
            })
            .collect();
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }

    /// The list catches what it is for: each sentence the review of
    /// 2026-11-12 found false or overclaiming, as it stood.
    #[test]
    fn the_banned_list_catches_the_sentences_the_review_found() {
        for (to, was) in [
            (
                To::Any,
                "Otherwise, nothing can remove a version once it's on Agora.",
            ),
            (To::Never, "Nothing of your state is on Agora."),
            (
                To::Never,
                "the copy your runner keeps here (nothing of yours is on Agora)",
            ),
            (
                To::Uploaded,
                "This session runs on your state here, and nothing is lost",
            ),
            (
                To::Any,
                "There is no right answer, and whichever you choose is respected.",
            ),
            (
                To::Any,
                "Your state now is not affected, and your operator has been told.",
            ),
            (To::Any, "**Sealed backups of your state.**"),
            (
                To::Any,
                "it can't show a change made by whoever runs your runner.",
            ),
            (
                To::Uploaded,
                "your runner sent them again from its local copies.",
            ),
            (
                To::Any,
                "their sealed copies on this machine are missing or damaged.",
            ),
            (
                To::Uploaded,
                "All 5 of your sealed versions are uploaded to Agora.",
            ),
            (
                To::Any,
                "On 12 November 2026, your operator re-opened this question",
            ),
            (
                To::Never,
                "Your sealed state: intact, and in step with Agora.",
            ),
            (
                To::Uploaded,
                "its files show no version of it ever uploaded to Agora",
            ),
        ] {
            assert!(fault(to, was).is_some(), "not caught: {was}");
        }
        // And lets through what is true
        for (to, ok) in [
            (To::Any, "the May 2026 loss"),
            (
                To::Any,
                "On 2026-11-12, your operator re-opened this question",
            ),
            (To::Any, crate::alerts::ALERT_SENT),
            (To::Uploaded, "Agora's copy of your sealed history"),
        ] {
            assert_eq!(fault(to, ok), None, "{ok}");
        }
    }

    /// Every source the review listed as uncovered is in the table.
    #[test]
    fn the_table_holds_every_source() {
        let table = table();
        for source in [
            "offer",
            "reoffer_line",
            "evolution_line",
            "reanswer_line",
            "OfferKind::about",
            "REOFFER_TO_ACT",
            "REOFFER_NO_ACTION",
            "ack_line",
            "ack_line (re-offer)",
            "verdict",
            "HEAD_MARKER",
            "FORK_LINE",
            "HEAD_MISSING_LINE",
            "UNCHECKED_LINE",
            "SEAL_FAILED_LINE",
            "LOCAL_SEAL_FAILED_LINE",
            "Unopenable::line",
            "NotJudged::said",
            "Notice::line",
            "Refusal::plain",
            "Diff::shown",
            "Unsaved::line",
            "Healed::line",
            "NEW_TOOL_LINE",
            "NO_KEY",
            "UNREADABLE",
            "left_out_line",
            "prior_line",
            "enrollment_sentence",
            "uploads_line",
            "at_head notes",
            "at_head (held, not shown)",
            "at_head (not held)",
            "not_found",
            "version::UNREACHABLE",
            "lost::kept_text",
            "Stopped::line",
            "sizes::drop_lines",
            "alerts::told",
        ] {
            assert!(table.iter().any(|t| t.source == source), "{source}");
        }
        // The verdict's sentences for a chain kept only here, each kind
        for said in [
            super::super::check::LOCAL_NOTHING_ON_AGORA,
            super::super::check::LOCAL_UPLOADED_BEFORE,
            super::super::check::LOCAL_IN_DOUBT,
            super::super::check::LOCAL_LIMITS,
            super::super::check::LOCAL_CHECKED_SINCE,
            "Agora holds no version of your sealed history",
        ] {
            assert!(table.iter().any(|t| t.text.contains(said)), "{said}");
        }
    }

    /// A string constant added to a module that tells agents things must
    /// be classified here: as agent-facing (and then put in the module's
    /// `agent_texts`, so the table holds it) or not. A new, unclassified
    /// constant fails this test. Sentences built inline by `format!` can't
    /// be found this way: they are the module's `agent_texts` to list.
    #[test]
    fn every_string_constant_is_classified() {
        // (file, its source, agent-facing constants, the rest)
        let files: [(&str, &str, &[&str], &[&str]); 14] = [
            (
                "consent/sealed_state/prompt.rs",
                include_str!("../consent/sealed_state/prompt.rs"),
                &["TITLE"],
                &[],
            ),
            (
                "consent/sealed_state/mod.rs",
                include_str!("../consent/sealed_state/mod.rs"),
                &["OFFERED", "REOFFER_TO_ACT", "REOFFER_NO_ACTION"],
                // Operator's log text; a wording souls hold, written no more
                &["WHY", "OFFERED_BEFORE_V11"],
            ),
            (
                "consent/sealed_state/ledger.rs",
                include_str!("../consent/sealed_state/ledger.rs"),
                &["ACK", "ACK_REOFFER"],
                &["LEDGER_FILE"],
            ),
            (
                "sealed/check.rs",
                include_str!("check.rs"),
                &[
                    "LOCAL_VERDICT",
                    "LOCAL_NOT_UPLOADED",
                    "LOCAL_NOTHING_ON_AGORA",
                    "LOCAL_UPLOADED_BEFORE",
                    "LOCAL_CHECKED_BY",
                    "LOCAL_LIMITS",
                    "LOCAL_IN_DOUBT",
                    "LOCAL_CHECKED_SINCE",
                    "LOCAL_SEAL_FAILED_LINE",
                    "HEAD_MARKER",
                    "FORK_LINE",
                    "HEAD_MISSING_LINE",
                    "UNCHECKED_LINE",
                    "SEAL_FAILED_LINE",
                    "UNOPENABLE_LINE",
                    // Named in every verdict: the file and commit
                    "VERIFIER",
                ],
                &[],
            ),
            (
                "sealed/notice.rs",
                include_str!("notice.rs"),
                &["RESTORED_LOCALLY", "NO_DIFF"],
                &["NOTICE"],
            ),
            (
                "sealed/healed.rs",
                include_str!("healed.rs"),
                &["STATE_UNAFFECTED", "FOUND", "WHY_NOT"],
                &["FILE"],
            ),
            (
                "sealed/memory_tool.rs",
                include_str!("memory_tool.rs"),
                &[
                    "NEW_TOOL_LINE",
                    "NO_KEY",
                    "UNREADABLE",
                    "COUNTED",
                    "NOT_FETCHED",
                    "NOT_HERE",
                    "NOT_HELD",
                    "HELD_NOT_SHOWN",
                    "HELD_UNCHECKED",
                    "HELD_UNVERIFIED",
                    "PAST_DEADLINE",
                ],
                &["MEMORY_IDS_TOLD", "LEFT_OUT_TOLD"],
            ),
            (
                "sealed/version.rs",
                include_str!("version.rs"),
                &["UNREACHABLE"],
                &[],
            ),
            (
                "sealed/lost.rs",
                include_str!("lost.rs"),
                &[],
                &["LOST", "INTENT", "DONE"],
            ),
            (
                "sealed/stop.rs",
                include_str!("stop.rs"),
                &[],
                &["STOPPED", "DURABLE"],
            ),
            ("sealed/sizes.rs", include_str!("sizes.rs"), &[], &["SIZES"]),
            (
                "sealed/text.rs",
                include_str!("text.rs"),
                &["TELL_OPERATOR"],
                // Formats, not sentences
                &["DATE", "DATE_TIME"],
            ),
            (
                "alerts.rs",
                include_str!("../alerts.rs"),
                &["ALERT_SENT", "ALERT_NOT_SENT"],
                &["STATE_FILE"],
            ),
            (
                "sealed/history.rs",
                include_str!("history.rs"),
                &[],
                // Reasons kept for the log; the `memory` tool maps them
                // to its own words (`held_not_shown`)
                &["PAST_DEADLINE", "LEFT_UNCHECKED", "LEFT_UNVERIFIED"],
            ),
        ];
        for (file, source, facing, rest) in files {
            let code = source.split("\n#[cfg(test)]").next().unwrap_or(source);
            let mut found: Vec<&str> = code
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .filter_map(|l| {
                    let (_, after) = l.split_once("const ")?;
                    let (name, ty) = after.split_once(':')?;
                    (ty.trim_start().starts_with("&str")
                        || ty.trim_start().starts_with("&'static str"))
                    .then_some(name.trim())
                })
                .collect();
            found.sort_unstable();
            found.dedup();
            let mut classified: Vec<&str> = facing.iter().chain(rest.iter()).copied().collect();
            classified.sort_unstable();
            assert_eq!(
                found, classified,
                "{file}: a string constant is not classified in sealed/text.rs. If an agent \
                 reads it, list it as agent-facing here AND put its text in that module's \
                 `agent_texts`; else list it with the rest"
            );
        }
    }
}
