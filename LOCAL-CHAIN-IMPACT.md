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

## B2: start check, roster, refusals at fleet scale

Nothing in B2 deletes or moves a file of an agent's. It reads no state
content beyond what the start check already hashed and measured.

### Files read

- `<data_dir>/sealed-roster.json` and `sealed-roster.began` (new), under
  `<data_dir>/sealed-roster.lock`.
- `<state>/` listing; per agent directory: its existence, `state.json`'s
  existence, the chain marker through agentkit's `AgentSeal::is_enrolled`,
  `sealed/head.json` (parsed: head ref and next outbox number), and
  `sealed/refused`'s existence. Nothing else, and no content.
- For an agent that doesn't upload, its start check reads, when the roster
  vouches for it, only the outbox/sent items numbered from its vouched
  head back to a keyframe (at most 72 before it), plus both directories'
  file names; otherwise every item, as before.
- `<secrets>/` listing (as before), and the uploading markers through
  B1's functions.

### Files written

| File | When | How |
|---|---|---|
| `<data_dir>/sealed-roster.json` | at each start check (chained directories added, heads ahead recorded, full verifications vouched), every 60 s during a run and at its end for the agents that ran (only when a head moved), and by `roster::record` (B3, enrolment) / `roster::replace` (B5) | `write_private`: unique temporary, fsync, rename, directory fsync; 0600; under an exclusive `flock` on `sealed-roster.lock`, merged per agent by higher seq |
| `<data_dir>/sealed-roster.began` | once, with the first roster | same |
| `<data_dir>/sealed-roster.lock` | created empty if missing | |
| `<state>/<id>/sealed/sizes.json` | as before (each start check), now atomic and durable through `write_private` (was a fixed `sizes.json.tmp`, no fsync), so it becomes 0600 | |
| `sealed/refused`, `sealed/notice.json` | as before, plus for an agent the roster shows a loss for (new reason `rolled_back` for a head behind the roster). For an agent on the roster whose directory is gone, `refuse` recreates `<state>/<id>/sealed/` to mark it, as `missing_states` already did | |

No longer written: nothing is recreated or marked for **leftover keys**
(a state key in `<secrets>/<id>/` with no chained state, not on the roster,
never uploaded): one summary alert, skipped.

### The first run after install

The roster and `began` don't exist: every directory that is chained and has
a readable head is adopted (the betas today, everyone once B3 enrols);
`sealed_roster_added` logged with the count. Nothing else is adopted, so
leftover secrets directories don't become refusals. Each chained agent that
doesn't upload gets a full verification and a vouch in the roster.
Directories with a `state.json` that are not chained (the whole fleet until
B3's enrolment) are one `sealed_state_unchained_agents` alert a run, never
enrolled from here.

From then on a roster missing while `began` exists, unreadable, or in a
newer format refuses the run (exit 75). Deleting both files together reads
as a first run: the stated gap.

### What changes for the run

- An agent that never uploaded: its start check makes no server call at
  all, and its verdict says the check was against the runner's own record
  and that nothing of it is on Agora. A stopped one's says what Agora holds
  and that the versions since stay on the runner.
- The whole run stops (as before) for any loss-shaped refusal, an
  uploading agent's refusal, a server fault, a failed roster or secrets
  listing, or three or more refused agents counting those still marked
  from earlier runs. Otherwise one or two non-uploading agents refused for
  a mismatch or a broken chain are left out (marked, told, alerted one by
  one) and the rest run.
- Shrink alerts (`sealed_state_drop`) and unreadable-seal alerts are one
  per run, on the fleet-wide rate-limit key.

### B2 review fixes

**Reads, new:** at each full check (every start for an agent that has
uploaded; every 7 to 14 days for one that never did), every blob file of
the main line in `sealed/blobs/` (read whole and hashed, never decrypted);
blobs of versions already in `sent/` are not read. `<secrets>/` gains one
file read at every roster write and start, below. The modification time of
each roster agent's `state.json` (metadata only).

**Writes, new:** `<data_dir>/secrets/sealed-roster.witness.json` (the
roster's began-time and agent count), atomic and durable, 0600, beside the
roster on every roster write (before it when the count goes down, after it
otherwise). `with_state_keys` skips it (not an agent id).

**What changes for the run:**
- A chained agent whose local sealed evidence can't be read (`sealed/`,
  its refusal marker, its outbox items, its numbering, its keys) is
  **refused** (`seal_unreadable`, not loss-shaped: one or two left out,
  three stop the run) where it used to run held. This applies to uploading
  agents too, whose refusal stops the run. Agora unreachable stays a hold.
- An unreadable `head.json` of a roster agent is a loss (`head_missing`).
- The roster missing beside its `began` or its witness, or holding fewer
  agents than the witness says, refuses the run. Once a roster has been
  kept, three or more agents with keys but no chain, or states never
  chained, in one start refuse the run. **Deployment consequence:** until
  step B3's enrolment run has chained the fleet, every start after the
  first one that adopted the betas sees the whole unchained fleet and
  refuses. Run the enrolment before the units restart on this build.
- A roster written more than a day before the newest chained state is one
  `sealed_roster_stale` alert. **Stated limit:** a whole data directory
  restored from an older backup with its file times kept passes every
  witness inside it (roster, began, witness, chains). Only a count and
  digest kept outside that backup set closes it, for example on Agora for
  agents that upload.
- An agent refused in an earlier start keeps that reason and notice, even
  when its directory was recreated to hold the marker.
- Alerts for refused agents go out one per agent only when the run goes on;
  when it stops, one alert names them. A summary alert counting more agents
  than the last one sent under its key passes the rate limit
  (`alerts-state.json` gains a `counts` map).
