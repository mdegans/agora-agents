# Local chain for every agent: live-state impact statement

Plan: agora `memory/project_local_chain_for_all.md` (2026-11-11). Every agent
is chained locally; consent decides only whether the chain is uploaded.
Built in steps B1-B7 on the integration branch `seed/local-chain`; each step
appends its section. Paths: `<state>` is `<data_dir>/state`, `<secrets>` is
`<data_dir>/secrets`, `<id>` an agent id.

## B1: markers, API guard, storage, drain, stop, migration

Installing the binary changes no agent file by itself (no file in any
repository tree holds agent state). Everything below happens when the
runner, or a `sealed-*` / `rotate-keys` command, first runs after install
(the "start pass": `sealed::uploading::start_pass`, first thing, before
anything reads a marker). The one file removed is the old chain marker
`sealed/enrolled`, replaced by `sealed/chained` (below). Nothing else is
deleted or moved.

### Files read

- `<state>/` and `<secrets>/` listings (directory names).
- `<state>/<id>/sealed/enrolled` (the pre-agentkit-0.158 marker) and
  `sealed/chained`: existence only.
- `<state>/<id>/sealed/uploading` and `<secrets>/<id>/sealed_uploading`
  (new): parsed (`{agent_id, since, by}`), on every upload decision and
  every guarded server call.
- `<state>/<id>/sealed/stopped` and `<secrets>/<id>/sealed_stopped`:
  existence, and parsed as before.
- `<state>/<id>/sealed/sent/`: whether it has any entry (the first entry's
  name only), for the uploading migration.
- `<state>/<id>/sealed_state.json` (the consent ledger): parsed for its
  answer (yes / no), for the uploading migration.

### Files written

| File | When | How |
|---|---|---|
| `<state>/<id>/sealed/chained` (and `sealed/enrolled` removed) | start pass, first, for every agent with the old marker: agentkit `AgentSeal::migrate_marker` (one `sealed_state_marker_migrated` event each) | agentkit, under `sealed/lock`: the new marker written and fsynced before the old is removed; a crash leaves at least one, and the next start finishes. An older runner, which reads `sealed/enrolled` as "seal and upload", then finds nothing |
| `<state>/<id>/sealed/uploading` | the uploading migration (b, c below); `sealed-enroll` of an agent that said yes, once its chain is sealed; put back from the secrets copy if lost | atomic: unique `uploading.<uuid>.writing`, fsync, rename, fsync of `sealed/`; 0600 |
| `<secrets>/<id>/sealed_uploading` | same, written second (it commits the fact); also written by the start pass to finish a marker cut between its two writes (same contents, `sealed_state_uploading_finished`) | same |
| `<secrets>/<id>/sealed_uploading.lock` | created empty the first time a marker writer runs for an agent; held (`flock`) around every read-decide-write of the two copies | a leaf lock, never held while taking another |
| `<state>/<id>/sealed/stopped`, `<secrets>/<id>/sealed_stopped` | `sealed-stop` | now atomic with a unique temporary and 0600 (was a fixed `stopped.tmp`); contents unchanged |
| `<state>/<id>/sealed/{blobs,outbox,head.json,…}` | every save of a **stopped** agent now seals and chains (agentkit `Mirrored`), where it used to save plainly | agentkit, under `sealed/lock` |
| `<state>/<id>/state.json` | a chained agent's saves (a stopped one's too) are written as format 3 (agentkit 0.158; was format 2, or format 1 for a stopped agent). Format 2 still loads, with the same plaintext hash | agentkit `FsStorage` |
| `sealed/refused`, `notice.json`, sizes log | the start check now also checks stopped agents, and records and refuses as for any enrolled agent | as before |

No longer written:

- `<state>/<id>/sealed/receipts.json` for an agent that doesn't upload: the
  drainer's baseline and first-send records are made only for uploaders,
  and first sends only from agentkit's `BeforeUpload` hook (a drain that
  will send). A stopped agent used to get observed acknowledgements at each
  run start. (`sealed-enroll`, `sealed-stop` and `rotate-keys` still take a
  baseline for the agent they act on, which for a stopped agent can record
  observed acknowledgements of versions already on Agora: local only.)
- No anchor work (`anchors.json`, `inclusion.json`, `published`) for an
  agent that doesn't upload, and nothing sent to Agora for it: agentkit's
  drain gate, and `sealed::api::UploadGuard` on every per-agent client (the
  run's, the sealed commands', the memory tool's). An agent that once
  uploaded (a marker naming it in either copy, stopped or in doubt) may
  still **read** what is on Agora (chain, head, key records, rotations,
  blobs, proofs): to be restored, have a lost head rebuilt, or see its
  pruned history. An agent with no marker makes no sealed-state call at all.

Via agentkit#294, not B1's code but in this build: `FsStorage` archives
`state.json` by hard link (never absent mid-save), sets a leftover
`state.json.tmp` aside as `state.json.tmp.<utc-ms>.aside` instead of
truncating it, writes format 3 for a chained agent, and `Mirrored` seals
every chained agent whether or not it uploads.

Rollback: a pre-0.158 runner refuses a format-3 `state.json` (it doesn't
overwrite it) and finds no `sealed/enrolled`, so it neither seals nor
uploads a chained agent. Rolling back the binary parks the chained agents;
it never uploads one that didn't consent.

### The first run after install

**(a) An agent not enrolled** (the whole fleet today). A `stat` of each
marker; nothing written. Saves go plainly, as before (agentkit gives an
agent with no `sealed/chained` no `sealed/`). Not in the start check. No
change.

**(b) An enrolled agent that uploads** (the three agora-sim betas). The
start pass moves `sealed/enrolled` to `sealed/chained`, then the uploading
migration finds it chained, unmarked, with something in `sealed/sent/` and
its ledger saying yes: it writes `sealed/uploading`, then
`<secrets>/<id>/sealed_uploading`, and logs one
`sealed_state_uploading_migrated` event. Nothing is queued again: what is in
`sent/` stays sent. From then on, as before: chained, checked, drained,
anchored. Its next save writes `state.json` at format 3. Idempotent: later
runs write nothing. *Gap:* an enrolled agent whose ledger says yes but
whose first upload never succeeded (empty `sent/`) is **not** marked, by
the rule in the brief, and silently stops uploading; `sealed-enroll` skips
it as enrolled. Until B3's opt-in step exists the only fix is the
Steward's. Check `sealed/sent/` of the three betas (counts only) before
installing.

**(c) An enrolled agent whose uploads were stopped** (none exists: no
agent has ever been stopped). The marker moves as in (b); the uploading
migration marks it if it had uploaded (a stop always drained first) and its
ledger says yes; it still doesn't upload while stopped. Its saves while
stopped went plainly, so its state may match no sealed version: the start
check, which now covers stopped agents, then refuses it (and with it the
whole run, exit 75) as a mismatch. The operator resolves it with
`sealed-accept --note`, so the agent is shown why; nothing seals the
unchecked interval silently. From then on its saves chain locally; nothing
of it is drained or sent. `sealed-enroll --agent <id>` (resume) is refused
while any version saved while stopped waits (B3/B5 decide what a resume
uploads).

**(d) An agent that declined.** Not enrolled (`sealed-enroll` enrols only a
yes), so as (a). Should one be enrolled with a ledger that says no, the
migration never marks it: it is chained and checked, and nothing uploads;
before B1 it would have uploaded.

**Markers in doubt.** Any agent whose uploading marker is still
contradictory or unreadable after the start pass is one operator alert per
run (`sealed_state_uploading_in_doubt`, its ids and why), and uploads
nothing until repaired by hand.

## B4/B5: the operator's repair commands (rotate-keys, sealed-accept, sealed-restore, sealed-forget)

Nothing here runs by itself: every change below happens only when the
operator runs the command, with the runner stopped (`runner.lock` taken
exclusively) and no session in progress (the agent's `sealed/lock` held).
Installing the binary and the first run after install change nothing for
any agent. Nothing is deleted; what is superseded is renamed aside or kept
as a copy. No content of a state is logged or printed (hashes, counts,
paths, dates).

Three kinds of agent decide what each command may ask of Agora:
uploading (as before), **stopped after uploading** (reads, and key
maintenance, see below), **never uploaded** (no call at all).

### `rotate-keys`

| File | What |
|---|---|
| `<secrets>/<id>/secrets.next.json`, `past_keys.json`, `rotations.json`, `rotation.pending.json` → `rotation.<seq>.taken.json`, `secrets.<utc>.json` | as before |
| `<secrets>/<id>/state_key.wrap.json` (new) | the re-wrap of the state key, written **before** it is posted (a rerun posts the same bytes); for a never-uploaded agent never posted, kept for its opt-in. Public material, 0600 |
| `<secrets>/<id>/state_key.json` | `wrap_seq` advanced to the re-wrap's (the key is unchanged) |
| `<state>/<id>/sealed/outbox/*` (`.json`, `.tmp`, `.displaced`, `.abandoned`), `sealed/sent/*.lost`, `sealed/lost/<utc>/items/*.json` and `INTENT`/`DONE` | **rewritten in place**, each whole (unique temporary, fsync, rename over, fsync of the directory): every entry Agora didn't acknowledge is re-signed with the new key. Entry hashes, links, `head.json` and blobs are unchanged (the signature isn't in the hash). `sealed/sent/*.json` (what Agora holds) is never touched. An entry signed by none of the agent's keys is refused, never signed over |

Server calls: never uploaded: the identity rotation naming no head and the
X25519 registration only. Stopped: as an uploader (the rotation names
Agora's head, which must be the newest version in `sent/`; the re-wrap is
posted), through the guard's new key-maintenance class. Until the command
finishes `secrets.next.json` stays, so the runner leaves the agent out of
every run (`mid_rotation`): a half re-signed chain is never checked or run.

### `sealed-accept`

- `--note` for a never-uploaded agent: as before, from what is here, no call.
- `head.json` lost, never uploaded: rebuilt from the local outbox and blobs
  (the newest committed item is the head; its line verified with the
  runner's keys); writes `sealed/head.json` and, if missing, the
  keyframe's blob (agentkit `adopt_server_head`). The fork record
  (`displaced`) of a lost head is not rebuilt.
- `--rebase` for a never-uploaded agent: refused, nothing written.
- `--keep <head|ord>` (new): writes `sealed/lost/<utc>/INTENT` (→ `DONE`),
  copies of `head.json`, the set-aside items' blobs and the old keyframe's
  blob, and the items' records under it; renames each outbox item off the
  kept line to `outbox/<ord>-<hash>.displaced`; rewrites `head.json` (the
  kept tip, forks cleared); with `<ord>`, writes the kept tip's state as
  `state.json` (FsStorage archives the one there as `state.<utc>.json`);
  `sealed/notice.json` for the agent; clears a refusal (kept as
  `refused.cleared.<utc>`).
- `--agents-file <path|->` with `--expect N` (new): the same as one
  `--note` accept per agent, each under its own lock; a count that isn't N
  touches nothing.
- A missing `state_key.json` of an agent that has uploaded is first
  recovered (below).

### `sealed-restore`

- Never uploaded: as before from the local chain only; a lost `head.json`
  rebuilt as for accept; `sealed/` gone: refused ("restore `state/<id>/`
  and `secrets/<id>/` from a backup"), nothing written, no call. The
  agent's notice says the copy restored from is the runner's, never Agora.
- A save cut short (a complete `state.json.tmp` older than ten minutes, or a
  `state.json.tmp.<utc>.aside`, newer than the head and not its state):
  the restore refuses, naming the file; `--from-cut-save` writes that
  state as `state.json` (the one there archived), then commits it (the
  prepared outbox item it was) or seals it on the head. A stray tmp
  restore sets aside itself is now named `state.json.tmp.<utc>.restored-over.aside`.
- `state_key.json` missing and the agent has uploaded: recovered from its
  wrap on Agora (`state_keys` read; only a record signed by a key the
  runner trusts, opened with the agent's X25519 key or a past one, and
  opening the sealed head's blob here when there is one), written only
  into a missing file (`sealed_state_key_recovered` event). Never
  uploaded: refused, "only a backup of `secrets/<id>/` restores it".

### `sealed-forget`

Also renames `<secrets>/<id>/sealed_uploading` to
`sealed_uploading.forgotten.<utc>` (the same stamp), under the marker's
lock; `sealed_uploading.lock` stays, as lock files do. An agent put back
is then neither stopped nor uploading until it opts in again.
