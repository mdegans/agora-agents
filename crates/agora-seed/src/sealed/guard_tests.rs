//! The read paths through a real [`UploadGuard`] over the runner's own
//! [`SealedEnrolment`]: an agent that once uploaded, now stopped, can still
//! be restored from, have its head rebuilt from, and read its pruned
//! history back from Agora; one that never uploaded makes no call at all.
//! Synthetic state only.

use std::path::Path;
use std::sync::Arc;

use agora_agentkit::ids::AgentId;
use agora_agentkit::reactor::{FsStorage, Storage as _};
use agora_agentkit::state::storage::AgentSeal;

use super::api::UploadGuard;
use super::api::fake::FakeApi;
use super::check::{self, Outcome};
use super::{KeyStore, SealedEnrolment, TestDir};

type Guard = UploadGuard<FakeApi, Arc<SealedEnrolment>>;

fn guard(root: &Path, keys: &KeyStore) -> Guard {
    UploadGuard::new(
        FakeApi::new(),
        Arc::new(SealedEnrolment::with_keys(root.join("state"), keys.clone())),
    )
}

/// Everything waiting uploaded: on the fake's chain, its blobs held there.
async fn on_agora(api: &FakeApi, dir: &Path, id: AgentId) {
    let seal = AgentSeal::new(dir, id);
    *api.chain.lock().unwrap() = Ok(check::local_entries(&seal).await.unwrap());
    for item in seal.outbox().await.unwrap() {
        seal.mark_sent(item.ord).await.unwrap();
    }
    for f in std::fs::read_dir(dir.join("sealed/blobs")).unwrap() {
        let f = f.unwrap();
        let sha = f.file_name().to_str().unwrap().parse().unwrap();
        api.blobs
            .lock()
            .unwrap()
            .insert(sha, std::fs::read(f.path()).unwrap());
    }
}

async fn stop(root: &Path, keys: &KeyStore, id: AgentId) {
    super::stop::stop(
        &root.join("state"),
        keys,
        id,
        "asked",
        Box::pin(async { Ok(()) }),
    )
    .await
    .unwrap();
}

/// No uploading marker in either copy: never uploaded.
fn unmark(dir: &Path, keys: &KeyStore, id: AgentId) {
    for p in [
        dir.join("sealed").join(super::uploading::UPLOADING),
        keys.agent_dir(id).join(super::uploading::DURABLE),
    ] {
        let _ = std::fs::remove_file(p);
    }
}

fn writes(api: &FakeApi) -> Vec<String> {
    api.calls
        .lock()
        .unwrap()
        .iter()
        .filter(|c| {
            [
                "post_state_key",
                "post_anchor",
                "publish_chosen",
                "rotate_key",
            ]
            .contains(&c.as_str())
        })
        .cloned()
        .collect()
}

/// `sealed-restore` of a stopped agent whose `sealed/` is lost: read back
/// from Agora (the marker's secrets copy says it once uploaded), and it
/// stays stopped; never uploaded, nothing is asked.
#[tokio::test]
async fn a_stopped_agent_is_restored_from_agora_and_a_never_uploaded_one_asks_nothing() {
    for uploaded in [true, false] {
        let root = TestDir::new("guard-restore");
        let (id, keys, dir) = check::tests::enrolled(&root).await;
        let api = guard(&root, &keys);
        on_agora(&api, &dir, id).await;
        if uploaded {
            stop(&root, &keys, id).await;
        } else {
            unmark(&dir, &keys, id);
        }
        std::fs::rename(dir.join("sealed"), root.join("sealed.lost")).unwrap();
        api.calls.lock().unwrap().clear();
        let done = super::restore::restore(&root.join("state"), &keys, &api, id, None, false).await;
        if uploaded {
            let done = done.unwrap();
            assert!(done.stopped, "still stopped");
            assert!(api.calls.lock().unwrap().iter().any(|c| c == "chain"));
            assert!(writes(&api).is_empty(), "{:?}", api.calls.lock().unwrap());
            assert!(!super::uploads(&dir, &keys, id));
        } else {
            let e = format!("{:#}", done.unwrap_err());
            assert!(e.contains("local only"), "{e}");
            assert!(api.calls.lock().unwrap().is_empty(), "nothing asked");
        }
    }
}

/// `sealed-accept` with `head.json` gone: rebuilt from Agora's chain for a
/// stopped agent; for one that never uploaded, refused with nothing asked.
#[tokio::test]
async fn a_stopped_agents_lost_head_is_rebuilt_from_agora_and_a_never_uploaded_one_asks_nothing() {
    for uploaded in [true, false] {
        let root = TestDir::new("guard-accept");
        let (id, keys, dir) = check::tests::enrolled(&root).await;
        let api = guard(&root, &keys);
        on_agora(&api, &dir, id).await;
        if uploaded {
            stop(&root, &keys, id).await;
        } else {
            unmark(&dir, &keys, id);
        }
        let sealed = dir.join("sealed");
        std::fs::rename(sealed.join("head.json"), sealed.join("head.json.lost")).unwrap();
        let local = agora_agentkit::state::storage::plaintext_hash(
            &FsStorage::new(root.join("state"))
                .load_raw(id)
                .await
                .unwrap()
                .value,
        );
        let Outcome::Refuse { reason, detail } =
            check::check(&dir, id, &local, Default::default(), &keys, &api).await
        else {
            panic!("not refused")
        };
        check::refuse(
            &dir,
            id,
            "tarn",
            reason,
            &detail,
            &crate::alerts::Alerter::off(),
        )
        .await;
        api.calls.lock().unwrap().clear();
        let done = super::accept::accept(&root.join("state"), &keys, &api, id, None).await;
        if uploaded {
            assert!(done.unwrap().adopted, "head.json rebuilt from Agora");
            assert!(writes(&api).is_empty(), "{:?}", api.calls.lock().unwrap());
        } else {
            assert!(done.is_err());
            assert!(api.calls.lock().unwrap().is_empty(), "nothing asked");
        }
    }
}

/// The `memory` tool's history of a stopped agent: its pruned versions
/// fetched back from Agora; never uploaded, nothing asked.
#[tokio::test]
async fn a_stopped_agents_pruned_versions_are_fetched_and_a_never_uploaded_one_asks_nothing() {
    use super::props::{History, Mode, build};
    for uploaded in [true, false] {
        let b = build(&History {
            saves: 3,
            upload: 9,
            rotate: false,
            mode: Mode::Plain,
            upload_rebased: false,
        })
        .await;
        if uploaded {
            stop(&b.root, &b.keys, b.id).await;
        } else {
            unmark(&b.dir, &b.keys, b.id);
        }
        b.seal().prune_blobs().await.unwrap();
        // Not strict: the never-uploaded case is refused, by design
        let api = guard(&b.root, &b.keys);
        *api.chain.lock().unwrap() = Ok(b.server());
        *api.blobs.lock().unwrap() = b.api.blobs.lock().unwrap().clone();
        *api.public_key.lock().unwrap() = Some(b.keys.signing(b.id).unwrap().verifying_key());
        let h = super::history::resolve(
            &b.state_dir(),
            &b.keys,
            &api,
            b.id,
            &mut super::history::HistoryCache::default(),
        )
        .await
        .unwrap();
        let calls = api.calls.lock().unwrap().clone();
        if uploaded {
            assert!(h.left_out.is_empty(), "{:?}", h.left_out);
            assert!(calls.iter().any(|c| c == "blob"), "fetched: {calls:?}");
        } else {
            // Its pruned versions stay listed, unopened (the history says
            // why); what matters here is that nothing was asked
            assert!(calls.is_empty(), "nothing asked: {calls:?}");
        }
        assert!(writes(&api).is_empty());
    }
}
