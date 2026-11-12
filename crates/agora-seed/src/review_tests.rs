//! Tests for the review round of 2026-11-12 (state loss, stranding and
//! wrongful fleet stops) that cross the start (`main.rs`) and the
//! operator's commands. Synthetic states, temp dirs, no network.

use std::collections::BTreeMap;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::FsStorage;

use super::*;
use crate::sealed::enroll::tests::{agent, leave_journal};
use crate::sealed::unchainable::{self, Class};

fn enroll_command(dry_run: bool) -> sealed::cli::Command {
    sealed::cli::Command::SealedEnroll {
        agents: Vec::new(),
        opt_in: false,
        dry_run,
        jobs: Some(2),
        note: None,
        resume: false,
        against_answer: false,
        begin_roster: false,
    }
}

async fn enroll(root: &std::path::Path, dry_run: bool) -> anyhow::Result<()> {
    sealed::cli::run(
        enroll_command(dry_run),
        root,
        url::Url::parse("http://127.0.0.1:9").unwrap(),
        None,
    )
    .await
}

/// A start over `ids`: what the check found, the pool it left, and the
/// agents skipped by stage.
async fn start_run(
    root: &std::path::Path,
    ids: &[AgentId],
    api: &sealed::api::fake::Guarded,
) -> (SealedRun, Vec<AgentId>, BTreeMap<AgentId, &'static str>) {
    let state_dir = root.join("state");
    let keys = sealed::KeyStore::new(root.join("secrets"));
    let loaded = load_states(&FsStorage::new(&state_dir), &state_dir)
        .await
        .unwrap();
    let mut pool: Vec<_> = ids
        .iter()
        .map(|id| (*id, crate::agent_selection_tests::pooled("tarn").1))
        .collect();
    let mut skipped = Vec::new();
    let run = sealed_start_check(
        &mut pool,
        &loaded.sealed,
        root,
        &state_dir,
        &keys,
        api,
        &alerts::Alerter::off(),
        false,
        &mut skipped,
        None,
        None,
    )
    .await;
    (
        run,
        pool.into_iter().map(|(id, _)| id).collect(),
        skipped.into_iter().map(|s| (s.id, s.stage)).collect(),
    )
}

/// [`start_run`], with what the refusal rule said.
async fn start(
    root: &std::path::Path,
    ids: &[AgentId],
    api: &sealed::api::fake::Guarded,
) -> (Vec<AgentId>, BTreeMap<AgentId, &'static str>, Result<()>) {
    let (run, pool, skipped) = start_run(root, ids, api).await;
    (pool, skipped, refuse_run(&run, &alerts::Alerter::off()))
}

async fn fleet(root: &std::path::Path, n: usize) -> Vec<AgentId> {
    let mut ids = Vec::new();
    for _ in 0..n {
        ids.push(sealed::check::tests::enrolled_local(root).await.0);
    }
    ids
}

fn sealed_dir(root: &std::path::Path, id: AgentId) -> std::path::PathBuf {
    root.join("state").join(id.to_string()).join("sealed")
}

/// B1, end to end. A fleet holding healthy agents, agents with a leftover
/// session journal, agents with no signing key and one whose state won't
/// load is enrolled, then started: the healthy and the journalled agents
/// are chained and run; the three that can't be chained are named by the
/// enrolment (which exits nonzero), recorded, and left out of the run
/// without parking it; repaired, a rerun chains them and they leave the
/// record. Before the fix the journalled agents were never chained, and
/// three unchained agents refused every start.
#[tokio::test]
async fn a_fleet_with_journalled_keyless_and_unloadable_agents_enrols_and_starts() {
    let root = sealed::TestDir::new("review-b1-fleet");
    let state_dir = root.join("state");
    let secrets = root.join("secrets");
    let mut healthy = Vec::new();
    for memories in [&["a"][..], &["a", "b"], &["a", "b", "c"]] {
        healthy.push(agent(&root, memories).await.0);
    }
    let mut journalled = BTreeMap::new();
    for _ in 0..3 {
        let id = agent(&root, &["j1", "j2"]).await.0;
        journalled.insert(id, leave_journal(&root, id).await);
    }
    let mut keyless = Vec::new();
    for _ in 0..2 {
        let id = agent(&root, &["k"]).await.0;
        std::fs::rename(
            secrets.join(id.to_string()),
            secrets.join(format!("{id}.aside")),
        )
        .unwrap();
        keyless.push(id);
    }
    let unloadable = agent(&root, &["u"]).await.0;
    std::fs::write(
        state_dir.join(unloadable.to_string()).join("state.json"),
        b"{ not json",
    )
    .unwrap();
    let all: Vec<AgentId> = healthy
        .iter()
        .chain(journalled.keys())
        .chain(&keyless)
        .chain(std::iter::once(&unloadable))
        .copied()
        .collect();
    let chained = |id: &AgentId| sealed::is_enrolled(&state_dir.join(id.to_string()));

    // A dry run says the same classes, and writes nothing
    let dry = enroll(&root, true).await.unwrap_err().to_string();
    assert!(dry.contains("3 agent(s) with a state"), "{dry}");
    assert!(!all.iter().any(chained));
    assert!(!root.join(unchainable::RECORD).exists());

    // The enrolment: nonzero, naming what is left and why
    let e = enroll(&root, false).await.unwrap_err().to_string();
    assert!(
        e.contains("3 agent(s) with a state left without a chain")
            && e.contains("3 can't be chained until repaired"),
        "{e}"
    );
    for id in healthy.iter().chain(journalled.keys()) {
        assert!(chained(id), "{id} chained");
    }
    for (id, before) in &journalled {
        let journal = FsStorage::new(&state_dir).journal(*id);
        assert_eq!(&std::fs::read(journal.path()).unwrap(), before);
    }
    let record = unchainable::load(&root).unwrap();
    assert_eq!(
        record
            .agents
            .iter()
            .map(|(id, e)| (*id, e.class))
            .collect::<BTreeMap<_, _>>(),
        keyless
            .iter()
            .map(|id| (*id, Class::NoSigningKey))
            .chain(std::iter::once((unloadable, Class::StateUnloadable)))
            .collect::<BTreeMap<_, _>>()
    );

    // The start: not refused; exactly the chained agents run, the three
    // recorded are left out (never run unchained)
    let api = sealed::api::fake::guarded(&root);
    for _ in 0..2 {
        let (pool, skipped, refused) = start(&root, &all, &api).await;
        refused.unwrap();
        let mut ran = pool.clone();
        ran.sort();
        let mut expected: Vec<AgentId> = healthy.iter().chain(journalled.keys()).copied().collect();
        expected.sort();
        assert_eq!(ran, expected);
        for id in keyless.iter().chain(std::iter::once(&unloadable)) {
            assert_eq!(skipped.get(id), Some(&"unchainable"), "{id}");
        }
    }
    assert!(
        api.calls.lock().unwrap().is_empty(),
        "nothing asked of Agora"
    );
    // The journals are still there for the sessions that will show them
    for (id, before) in &journalled {
        let journal = FsStorage::new(&state_dir).journal(*id);
        assert_eq!(&std::fs::read(journal.path()).unwrap(), before);
        assert!(interrupted::notice(&journal.read().await.unwrap()).is_some());
    }

    // Repaired (the keys put back), a rerun chains them; the unloadable
    // one stays recorded and the command still exits nonzero for it
    for id in &keyless {
        std::fs::rename(
            secrets.join(format!("{id}.aside")),
            secrets.join(id.to_string()),
        )
        .unwrap();
    }
    let e = enroll(&root, false).await.unwrap_err().to_string();
    assert!(e.contains("1 agent(s) with a state left"), "{e}");
    assert!(keyless.iter().all(chained));
    let record = unchainable::load(&root).unwrap();
    assert_eq!(
        record.agents.keys().copied().collect::<Vec<_>>(),
        vec![unloadable]
    );
    let (pool, skipped, refused) = start(&root, &all, &api).await;
    refused.unwrap();
    assert_eq!(pool.len(), all.len() - 1);
    assert_eq!(skipped.get(&unloadable), Some(&"unchainable"));
}

/// The record excuses only the agents enrolment wrote into it: three
/// unchained states it doesn't name still refuse the run, recorded ones
/// beside them or not, and an unreadable record excuses nobody.
#[tokio::test]
async fn only_recorded_agents_are_excused_from_the_strays_rule() {
    let root = sealed::TestDir::new("review-b1-strays");
    let state_dir = root.join("state");
    let (chained, keys, _) = agent(&root, &["a"]).await;
    sealed::enroll::tests::enroller(&root, &keys)
        .chain(chained, false)
        .await
        .unwrap();
    let mut unchained = Vec::new();
    for _ in 0..3 {
        unchained.push(agent(&root, &["b"]).await.0);
    }
    let all: Vec<AgentId> = std::iter::once(chained).chain(unchained.clone()).collect();
    let api = sealed::api::fake::guarded(&root);
    // Two recorded, one not: one stray, the run goes on, two left out
    unchainable::settle(
        &root,
        unchained[..2]
            .iter()
            .map(|id| (*id, unchainable::Found::Cannot(Class::NoSigningKey))),
    )
    .unwrap();
    let (pool, skipped, refused) = start(&root, &all, &api).await;
    refused.unwrap();
    assert_eq!(skipped.values().filter(|s| **s == "unchainable").count(), 2);
    assert!(pool.contains(&chained) && pool.contains(&unchained[2]));
    // Three more nobody recorded: refused
    for _ in 0..3 {
        agent(&root, &["c"]).await;
    }
    let (_, _, refused) = start(&root, &all, &api).await;
    assert!(refused.unwrap_err().is::<RefusedRun>());
    // An unreadable record excuses nobody: the first three are strays too
    let root2 = sealed::TestDir::new("review-b1-strays-unreadable");
    let (chained, keys, _) = agent(&root2, &["a"]).await;
    sealed::enroll::tests::enroller(&root2, &keys)
        .chain(chained, false)
        .await
        .unwrap();
    let mut ids = vec![chained];
    for _ in 0..3 {
        ids.push(agent(&root2, &["b"]).await.0);
    }
    std::fs::write(root2.join(unchainable::RECORD), b"{").unwrap();
    let api = sealed::api::fake::guarded(&root2);
    let (_, _, refused) = start(&root2, &ids, &api).await;
    assert!(refused.unwrap_err().is::<RefusedRun>());
    let _ = state_dir;
}

/// S5. Sealed history that can't be read for a moment (here, an outbox
/// item torn and then put back) refuses the agent for that start and
/// marks nothing: the next start judges it afresh and it runs, with no
/// `sealed-accept` and no notice. It used to be marked, and the marker
/// answered "still refused" before the check looked again.
#[tokio::test]
async fn an_unreadable_seal_is_not_marked_and_is_judged_again() {
    let root = sealed::TestDir::new("review-s5-unreadable");
    let ids = fleet(&root, 3).await;
    let api = sealed::api::fake::guarded(&root);
    let item = std::fs::read_dir(sealed_dir(&root, ids[0]).join("outbox"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "json"))
        .unwrap();
    let good = std::fs::read(&item).unwrap();
    std::fs::write(&item, b"{ torn").unwrap();
    let (run, pool, _) = start_run(&root, &ids, &api).await;
    assert_eq!(
        run.refused
            .iter()
            .map(|r| (r.id, r.reason))
            .collect::<Vec<_>>(),
        vec![(ids[0], sealed::notice::Refusal::SealUnreadable)]
    );
    assert_eq!(pool.len(), 2, "left out of this run");
    refuse_run(&run, &alerts::Alerter::off()).unwrap();
    assert!(!sealed_dir(&root, ids[0]).join("refused").exists());
    assert!(
        sealed::notice::Notice::load(&root.join("state").join(ids[0].to_string()))
            .unwrap()
            .is_none(),
        "nothing to tell the agent"
    );
    // What the operator is told is true: nothing marked, checked again
    let options = refusal_options(ids[0], sealed::notice::Refusal::SealUnreadable);
    assert!(
        options[0].contains("the next start checks again"),
        "{options:?}"
    );
    assert!(
        options
            .iter()
            .all(|o| !o.contains("sealed-accept") && !o.contains("sealed-restore")),
        "{options:?}"
    );
    std::fs::write(&item, good).unwrap();
    let (run, pool, _) = start_run(&root, &ids, &api).await;
    assert!(run.refused.is_empty(), "{:?}", run.refused);
    assert_eq!((pool.len(), run.outstanding), (3, 0));
}

/// S5. When one of the runner's own records fails (here the roster can't
/// be read), the run is refused and no agent is checked or marked: a
/// state that would have been refused on its own is judged by the start
/// that can read its records. Every agent used to be checked and marked
/// all the same, each then needing an accept and a notice.
#[tokio::test]
async fn a_fleet_level_failure_marks_no_agent() {
    let root = sealed::TestDir::new("review-s5-fleet");
    let ids = fleet(&root, 3).await;
    let api = sealed::api::fake::guarded(&root);
    start(&root, &ids, &api).await.2.unwrap();
    for id in &ids {
        FsStorage::new(root.join("state"))
            .save_raw(*id, serde_json::json!({ "memory": "" }), None)
            .await
            .unwrap();
    }
    let roster = root.join(sealed::roster::ROSTER);
    let good = std::fs::read(&roster).unwrap();
    std::fs::write(&roster, b"{ torn").unwrap();
    let (run, _, _) = start_run(&root, &ids, &api).await;
    assert_eq!(run.fleet.len(), 1, "{:?}", run.fleet);
    assert!(run.fleet[0].contains("Way out"), "{:?}", run.fleet);
    assert!(run.refused.is_empty(), "{:?}", run.refused);
    assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
    for id in &ids {
        assert!(!sealed_dir(&root, *id).join("refused").exists(), "{id}");
    }
    // The record readable again: now each is judged, and marked
    std::fs::write(&roster, good).unwrap();
    let (run, _, _) = start_run(&root, &ids, &api).await;
    assert_eq!(run.refused.len(), 3);
    assert!(sealed_dir(&root, ids[0]).join("refused").exists());
}

/// S8, at the start. The roster, `began` and the witness all gone beside
/// chained agents (and their `sealed/` emptied of nothing: the directories
/// are as they were) used to read as a first run: every directory adopted
/// as found, the strays rule off. The run is refused, the refusal names
/// the one command, and after the operator's deliberate begin the start
/// runs.
#[tokio::test]
async fn a_start_never_begins_a_roster_over_chained_agents() {
    let root = sealed::TestDir::new("review-s8");
    let ids = fleet(&root, 4).await;
    let api = sealed::api::fake::guarded(&root);
    start(&root, &ids, &api).await.2.unwrap();
    for (n, file) in [
        root.join(sealed::roster::ROSTER),
        root.join(sealed::roster::BEGAN),
        root.join("secrets").join(sealed::roster::WITNESS),
    ]
    .into_iter()
    .enumerate()
    {
        std::fs::rename(&file, root.join(format!("lost-{n}"))).unwrap();
    }
    let (run, _, _) = start_run(&root, &ids, &api).await;
    assert_eq!(run.fleet.len(), 1, "{:?}", run.fleet);
    assert!(
        run.fleet[0].contains("4 chained agent") && run.fleet[0].contains("--begin-roster --note"),
        "{:?}",
        run.fleet
    );
    assert!(run.refused.is_empty(), "no agent marked for it");
    let e = refuse_run(&run, &alerts::Alerter::off()).unwrap_err();
    assert!(e.is::<RefusedRun>());
    assert!(!root.join(sealed::roster::ROSTER).exists(), "nothing begun");
    // Plain enrolment doesn't begin one over chained agents either
    let e = enroll(&root, false).await.unwrap_err().to_string();
    assert!(e.contains("--begin-roster --note"), "{e}");
    // The deliberate begin, with its note
    let mut begin = enroll_command(false);
    if let sealed::cli::Command::SealedEnroll {
        begin_roster, note, ..
    } = &mut begin
    {
        *begin_roster = true;
        *note = Some("the data dir's top level was restored without it".to_string());
    }
    sealed::cli::run(
        begin,
        &root,
        url::Url::parse("http://127.0.0.1:9").unwrap(),
        None,
    )
    .await
    .unwrap();
    assert!(
        std::fs::read_to_string(root.join(sealed::roster::BEGAN))
            .unwrap()
            .contains("restored without it")
    );
    let (pool, _, refused) = start(&root, &ids, &api).await;
    refused.unwrap();
    assert_eq!(pool.len(), 4);
    assert!(api.calls.lock().unwrap().is_empty());
}

/// An agent left out alone for a mismatch whose state is deleted
/// afterwards has suffered a loss since: the next start judges it as one
/// (it used to keep the marker's `mismatch`, which never stops the run),
/// and the marker follows.
#[tokio::test]
async fn a_loss_after_a_mismatch_is_judged_as_the_loss() {
    use sealed::notice::Refusal;
    let root = sealed::TestDir::new("review-rejudged");
    let ids = fleet(&root, 3).await;
    let api = sealed::api::fake::guarded(&root);
    let dir = root.join("state").join(ids[0].to_string());
    FsStorage::new(root.join("state"))
        .save_raw(ids[0], serde_json::json!({ "memory": "" }), None)
        .await
        .unwrap();
    let (run, pool, _) = start_run(&root, &ids, &api).await;
    assert_eq!(run.refused[0].reason, Refusal::Mismatch);
    refuse_run(&run, &alerts::Alerter::off()).unwrap();
    assert_eq!(pool.len(), 2);
    // Its state deleted since
    std::fs::rename(dir.join("state.json"), root.join("gone.json")).unwrap();
    let (run, _, _) = start_run(&root, &ids, &api).await;
    assert_eq!(
        run.refused
            .iter()
            .map(|r| (r.id, r.reason))
            .collect::<Vec<_>>(),
        vec![(ids[0], Refusal::StateMissing)]
    );
    assert!(refuse_run(&run, &alerts::Alerter::off()).is_err());
    assert_eq!(
        sealed::check::marked_reason(&dir),
        Some(Refusal::StateMissing)
    );
    // The other way never: a loss keeps its reason
    assert_eq!(
        sealed::check::rejudged(&dir, Refusal::Mismatch),
        Refusal::StateMissing
    );
}

/// A reactor that ended with an error is alerted, on its endpoint's key,
/// apart from the run's summary of unsaved sessions.
#[tokio::test]
async fn a_failed_reactor_is_alerted() {
    let root = sealed::TestDir::new("review-reactor-failed");
    let sent = alerts::testing::Recorder::default();
    let alerter = alerts::testing::recording(root.path(), &sent);
    alerter.notify(reactor_failed_alert(
        "http://localhost:8080",
        "storage error",
    ));
    alerter.flush(Duration::from_secs(5)).await;
    let mails = sent.sent();
    assert_eq!(mails.len(), 1);
    assert!(
        mails[0].contains("state_unsaved") && mails[0].contains("ended with an error"),
        "{}",
        mails[0]
    );
}

mod reload {
    use super::*;
    use agora_agentkit::state::storage::{AgentSeal, Mirrored};

    type TestStorage = Mirrored<FsStorage, (), Arc<sealed::SealedEnrolment>>;

    fn storage(root: &std::path::Path) -> TestStorage {
        Mirrored::new(
            FsStorage::new(root.join("state")),
            (),
            Arc::new(sealed::SealedEnrolment::new(
                root.join("state"),
                root.join("secrets"),
            )),
        )
    }

    fn agent_state(
        identity: &str,
        last: Option<chrono::DateTime<chrono::Utc>>,
    ) -> serde_json::Value {
        let (_, mut tarn) = crate::agent_selection_tests::pooled("tarn");
        tarn.last_cycle_at = last;
        let mut v = serde_json::to_value(&tarn).unwrap();
        v["soul"]["identity"] = serde_json::json!(identity);
        v
    }

    async fn chained(root: &std::path::Path) -> (AgentId, sealed::KeyStore, PathBuf) {
        let (id, keys, dir) = sealed::check::tests::enrolled_local(root).await;
        sealed::props::save(root, id, agent_state("An agent.", None)).await;
        (id, keys, dir)
    }

    fn cohort(states: &[(AgentId, SeedState)]) -> Vec<BTreeMap<String, Vec<(AgentId, SeedState)>>> {
        let twins = states
            .iter()
            .map(|(id, s)| {
                (
                    *id,
                    serde_json::from_value(serde_json::to_value(s).unwrap()).unwrap(),
                )
            })
            .collect();
        vec![BTreeMap::from([("gpt-oss-120b.gguf".to_string(), twins)])]
    }

    /// What a claim over `root`'s one agent came to, after `change` ran
    /// between the load and the claim.
    struct Claimed1 {
        routed: usize,
        run: SealedRun,
        skipped: Vec<(AgentId, &'static str)>,
        mails: Vec<String>,
    }

    async fn claim_after(
        root: &sealed::TestDir,
        keys: &sealed::KeyStore,
        min_cycle: Option<u64>,
        change: impl AsyncFnOnce(),
    ) -> Claimed1 {
        let state_dir = root.join("state");
        let storage = storage(root);
        let loaded = load_states(&storage, &state_dir).await.unwrap();
        change().await;
        let api = sealed::api::fake::guarded(root);
        let sent = alerts::testing::Recorder::default();
        let alerter = alerts::testing::recording(root.path(), &sent);
        let was_chained: BTreeSet<AgentId> = loaded.sealed.keys().copied().collect();
        let reload = Reload {
            storage: &storage,
            check: StartCheck {
                data_dir: root.path(),
                state_dir: &state_dir,
                keys,
                api: &api,
                alerts: &alerter,
                start: None,
            },
            chained: &was_chained,
            min_cycle: &|_, _, _| min_cycle,
        };
        let mut run = SealedRun::default();
        let mut skipped = Vec::new();
        let mut cohorts = cohort(&loaded.states);
        let claimed = claim_sessions(
            &mut cohorts,
            &loaded.seen,
            None,
            &reload,
            &mut run,
            &mut loaded.bases.clone(),
            &mut skipped,
            &alerter,
        )
        .await;
        drop(claimed);
        alerter.flush(Duration::from_secs(5)).await;
        Claimed1 {
            routed: cohorts
                .iter()
                .map(|c| c.values().map(Vec::len).sum::<usize>())
                .sum(),
            run,
            skipped: skipped.into_iter().map(|s| (s.id, s.stage)).collect(),
            mails: sent.sent(),
        }
    }

    /// S7. An agent read again under its session lock gets the start's own
    /// check: a state that matches nothing is refused there as at the
    /// start, marked, and **alerted** (the refusal rule has already run);
    /// it used to be left out with nothing marked and no alert.
    #[tokio::test]
    async fn a_refusal_on_reload_is_marked_and_alerted() {
        let root = sealed::TestDir::new("review-s7-refused");
        let (id, keys, dir) = chained(&root).await;
        let out = claim_after(&root, &keys, None, async || {
            FsStorage::new(root.join("state"))
                .save_raw(id, agent_state("Edited by hand.", None), None)
                .await
                .unwrap();
        })
        .await;
        assert_eq!(out.routed, 0);
        assert_eq!(out.skipped, vec![(id, "sealed")]);
        assert_eq!(out.run.refused.len(), 1);
        assert!(AgentSeal::new(&dir, id).is_refused().await.unwrap());
        assert!(
            out.mails.iter().any(|m| m.contains("sealed_state_refused")),
            "{:?}",
            out.mails
        );
    }

    /// S7. Chained when loaded, its chain marker gone by the time it is
    /// claimed: refused as a missing head and alerted. It used to skip the
    /// check and run, saving outside its chain.
    #[tokio::test]
    async fn a_marker_gone_since_the_load_is_a_refusal_not_a_plain_session() {
        let root = sealed::TestDir::new("review-s7-marker");
        let (id, keys, dir) = chained(&root).await;
        let out = claim_after(&root, &keys, None, async || {
            sealed::props::save(&root, id, agent_state("Changed elsewhere.", None)).await;
            std::fs::rename(
                dir.join("sealed")
                    .join(agora_agentkit::state::storage::CHAINED),
                root.join("marker.aside"),
            )
            .unwrap();
        })
        .await;
        assert_eq!(out.routed, 0, "never run unchained");
        assert_eq!(
            out.run
                .refused
                .iter()
                .map(|r| (r.id, r.reason))
                .collect::<Vec<_>>(),
            vec![(id, sealed::notice::Refusal::HeadMissing)]
        );
        assert!(dir.join("sealed").join("refused").exists());
        assert!(
            out.mails.iter().any(|m| m.contains("sealed_state_refused")),
            "{:?}",
            out.mails
        );
    }

    /// S7. An agent another process gave a session since this run routed
    /// it is not run again inside `min_cycle_secs`; with no cadence gate
    /// (or once due) it runs from its new state, seated and queued to
    /// drain as the start seats any agent.
    #[tokio::test]
    async fn a_session_elsewhere_counts_for_the_cadence_gate() {
        let root = sealed::TestDir::new("review-s7-due");
        let (id, keys, dir) = chained(&root).await;
        let now = chrono::Utc::now();
        let ran = async || {
            sealed::props::save(
                &root,
                id,
                agent_state("Ran elsewhere.", Some(chrono::Utc::now())),
            )
            .await;
        };
        let out = claim_after(&root, &keys, Some(3600), ran).await;
        assert_eq!(out.routed, 0, "not due again");
        assert!(out.skipped.is_empty(), "not an error: {:?}", out.skipped);
        assert!(out.run.refused.is_empty() && out.mails.is_empty());
        assert!(!AgentSeal::new(&dir, id).is_refused().await.unwrap());
        let out = claim_after(&root, &keys, None, ran).await;
        assert_eq!(out.routed, 1);
        assert!(out.run.seats.contains_key(&id));
        assert!(out.run.drained.iter().any(|(a, _)| *a == id));
        assert!(due_after(Some(3600), Some(now), now).is_some());
        assert!(due_after(Some(3600), Some(now - chrono::Duration::hours(2)), now).is_none());
        assert!(due_after(None, Some(now), now).is_none());
    }
}
