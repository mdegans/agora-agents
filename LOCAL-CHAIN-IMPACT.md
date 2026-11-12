# Local chain for every agent: live-state impact statement

Plan: agora `memory/project_local_chain_for_all.md` (2026-11-11). Every agent
is chained locally; consent decides only whether the chain is uploaded.
Built in steps B1-B7 on `seed/local-chain` and integrated there
(B1-B5 on 2026-11-11; B6, B7 and the last section on 2026-11-12). Paths: `<state>` is
`<data_dir>/state`, `<secrets>` is `<data_dir>/secrets`, `<id>` an agent id.

The first part is the integrated statement: what the first start after
install and the fleet's enrolment run do to each kind of agent, the start
and deploy orders, the one-way door and the stated limits. The step
sections that follow are each builder's, kept whole, with a note where
integration changed what they say (marked **Integration:**).

## Integrated: what changes, per kind of agent

Nothing in this build deletes a file of an agent's. Superseded files are
renamed aside or kept as copies. Two files are removed by design: the old
chain marker `sealed/enrolled`, once `sealed/chained` is written and
fsynced (agentkit `migrate_marker`), and the `memory` tool's search index
when `sealed-forget` runs (a sidecar rebuilt on demand). Installing the
binary changes nothing by itself. Everything below happens when the runner
or a `sealed-*` / `rotate-keys` command first runs after install.

**Start order** (`main.rs` states it in one comment): `promote` (a save cut
between its archive and its rename, only while no other runner shares the
data dir) → `runner.lock` shared → the uploading start pass (the chain
marker moved first, uploading markers repaired, the one-time migration,
markers in doubt alerted) → the stop reconcile → `load_states` (finishes a
sealed commit a crash cut short) → the governance log verified → the
roster and its witnesses → a pre-loss copy's pair cut short finished, or a
copy the agent chose to see appended → each agent's start check → the
refusal rule → reactors. Every operator command that writes takes
`runner.lock` exclusively **before** its start pass and holds it to the
end (`sealed-enroll` but `--dry-run`, `sealed-reoffer`, `sealed-accept`,
`sealed-restore`, `sealed-forget`, `rotate-keys` but `--status`);
`sealed-stop` runs beside a runner.

**Data-directory files, once:** `<data_dir>/sealed-uploading-migrated`
(the one-time migration's record), `<data_dir>/sealed-roster.json`,
`sealed-roster.began`, `sealed-roster.lock`, and
`<secrets>/sealed-roster.witness.json` (the roster's began-time and agent
count; `with_state_keys` skips it). All atomic and durable, 0600.

### Unenrolled today (the fleet, about 1,663 agents)

*First start after install.* Read: `<state>/` and `<secrets>/` listings,
each agent's `sealed/enrolled`, `sealed/chained`, uploading and stop
markers (existence; none exist), `state.json` (loaded as before). Written:
the data-directory files above (the roster begun, adopting only the agents
already chained, with a state), nothing in an agent's directory beyond its
ordinary saves. Saves stay plain (no `sealed/`); FsStorage now archives
`state.json` by hard link and sets a leftover `state.json.tmp` aside as
`state.json.tmp.<utc-ms>.aside`. One `sealed_state_unchained_agents` alert
gives the count. **The second start refuses the run** (three or more
agents with a state and no chain beside a kept roster), and the refusal
says to run `agora-seed sealed-enroll` with the runner stopped. So: enrol
before the units start (deploy order, below).

*The fleet's enrolment run* (`agora-seed sealed-enroll`, runner stopped;
no network call). Per agent, read: `state.json`, every `state.<utc>.json`
archive (state and mtime), the memory-restore ledger and recovered copy
(an agent that chose `show`), the session journal's existence (an agent
with one is skipped), its signing key. Written:
`<secrets>/<id>/state_key.json` (new, cached, never posted),
`<state>/<id>/sealed/` (genesis, the pre-loss copy if chosen, one entry
and blob per distinct archive, the current state: `blobs/`, `outbox/`,
`head.json`), the sizes sidecar, `sealed/left_out.json` (archives that can
never be read: name and reason class only), `state.json` rewritten at
format 3 (the previous one hard-linked as an archive first), then
`sealed/chained`; the roster records its head (the witness's count rises).
A run cut short is finished by the next. Nothing uploads. The next start
finds every agent on the roster and verifies each one's whole local line
(then the tail only, with a full check every 7-14 days); its verdict says
the check was against the runner's own record and that nothing of it is
on Agora.

### Enrolled and uploading (the three test agents)

*First start.* `sealed/enrolled` → `sealed/chained` (agentkit, mtime and
bytes kept). The one-time uploading migration finds each chained, with
`sent/` non-empty and its ledger's latest answer yes: writes
`sealed/uploading`, then `<secrets>/<id>/sealed_uploading` (and its
`.lock`); then `sealed-uploading-migrated` is written and the migration
never runs again. Adopted into the roster. The full start check against
Agora, as before. Its next save writes `state.json` at format 3. agentkit
0.158 **keeps acknowledged blobs** (`remove_unnamed_blobs` replaces
`prune_blobs`), so `sealed/blobs/` stops shrinking after uploads. Each
drain reads Agora's head (`Unchecked` when it can't, which holds nothing)
and Agora's report of damaged blobs: agentkit writes `sealed/healing.json`
(the last damaged list's count and time) and sends back the blobs it
holds; the runner sums each pass (`sealed_state_healing`) and alerts
`sealed_state_agora_blobs_unhealable` when blobs Agora lost can't be sent
again from here.

*The enrolment run.* Already chained: nothing, unless it chose `show`
since it was chained (then the pre-loss copy is appended and uploaded, and
it is told). The same append now also happens at run start.

### Declined

An agent whose latest answer is `decline` (or a legacy `refused`) and was
never enrolled: as "unenrolled" above; the enrolment run chains it; it
never uploads; the offer isn't put again (its answer is on file) unless
the operator re-opens it (`sealed-reoffer`). One enrolled under an older
runner whose ledger now says no: its marker moves; the migration doesn't
mark it; it is chained and checked locally, and nothing of it is sent or
read. **Check before installing** (counts only): no agent with a non-empty
`sealed/sent/` and a ledger that isn't yes, since such an agent would read
as never uploaded (its verdict would say nothing of it is on Agora).

### Stopped (none exists)

Marker moved as above; marked uploading by the migration if it had
uploaded and its ledger says yes; still not uploading while stopped. Its
saves now chain locally. A stop from before the chain went on while
stopped would be refused at its first start check (its saves went plain);
the operator resolves it with `sealed-accept --note`. `sealed-enroll
--resume --agent <id>` resumes: the versions saved while stopped are set
aside (`.displaced`, kept, never uploaded) and one new snapshot is sealed
on the uploaded line.

### Forgotten (`sealed-forget` was run)

Its `state.json` is gone; its state key, memory-id key, durable stop and
durable uploading marker are renamed `*.forgotten.<utc>*` with a note.
`sealed-forget` now also takes it off the roster (`roster::replace(None)`,
the witness's count lowered first), and its leftover `sealed/refused` no
longer counts toward the refusal threshold. A chained directory left
without a state is never adopted into the roster again. Put back (its
directory and state key copied back), the start pass renames its old local
markers `sealed/<name>.forgotten.<utc>`: it is neither uploading nor
stopped until an explicit opt-in, and is adopted into the roster at the
next start.

## Integrated: deploy order

1. Server first (migration 149 and the hardening branch).
2. Release agentkit 0.158.0; drop the TEMPORARY path patch (the workspace
   already asks for 0.158.0, and `VERIFIER` names it).
3. The dry run on a `cp -a` copy of `state/` **and** `secrets/`, fleet
   metrics before and after (counts and sizes only); count secrets
   directories holding a state key against chained directories, and check
   for the declined agents above.
4. Stop both units and the timer.
5. Install the runner.
6. `agora-seed sealed-enroll --jobs 16`, before any start. It runs the
   start pass itself (marker move, migration) and begins the roster.
   Estimated 7-12 minutes (synthetic: about 0.3 s a version).
7. Fleet metrics again; start the units; compare after the first sweep.

If a unit starts before step 6, the first start runs (adopting the test
agents and alerting the unchained count) and every later start refuses,
naming `sealed-enroll`, which that refusal doesn't block. Integration
tests that sequence.

## Integrated: the one-way door

Once the start pass moves `sealed/enrolled` to `sealed/chained`, or a
chained agent's first save writes `state.json` at format 3, a runner
older than this build finds no `sealed/enrolled` and refuses a format-3
state: it parks the agent; it never seals it plainly or uploads it. After
the enrolment run that is the whole fleet: **rolling the binary back parks
every agent**; it never uploads one that didn't consent. A ledger that
`sealed-reoffer` wrote is format 3 too, which an older runner refuses
rather than drop the re-offer.

## Integrated: stated limits

- A local chain catches accidents (the May loss's shape: a state replaced,
  a directory rolled back or removed). It proves nothing against the
  operator, who holds the disk and the keys; agent-facing text says the
  check was against the runner's own record.
- A whole data directory restored from an older backup with its file times
  kept passes every witness inside it (roster, `began`, the secrets
  witness, the chains). A roster written a day before the newest state is
  one alert. Deleting the roster, `began` and the witness together reads as
  a first run.
- A state lost before the roster first recorded its agent is found only by
  its keys (`missing_states`), not by the roster: a chained directory
  without a state is not adopted.
- An agent the roster refuses for a missing directory has
  `<state>/<id>/sealed/` recreated to hold its marker (B2, unchanged).
- `sealed-accept` of a rolled-back agent cut after its new head is written
  but before the roster takes it: the next start refuses it as rolled back
  again; accepting again resolves it.
- An agent that uploaded but has no uploading marker reads as never
  uploaded (see Declined).
- A pre-loss copy chosen after chaining is looked for at every start for
  the agents whose memory-restore record says `show`: their chain's items
  are read each start until the copy is in.

## Integrated: what integration changed (beyond merging)

- **The roster set by the operator's commands.** `sealed-accept` (accept,
  `--rebase`, a loss resolved, `--keep`) and `sealed-restore` (a version, a
  cut save) call `roster::replace(Some(head))` after `head.json` is
  written; `sealed-forget` calls `replace(None)` after its commit (a rerun
  cut before it finishes that too). `replace(None)` begins no roster.
- **Adoption needs a state.** The roster adopts a chained directory only
  with a `state.json` (B2 adopted any readable head, which re-added a
  forgotten agent and refused it as missing a start later).
- **Wide accept.** `--agents-file` holds back every loss-shaped refusal by
  the start check's own `Refusal::loss_shaped`: `head_missing` and
  `rolled_back` now join state missing, unreadable, I/O and too new (B4/5's
  separate list removed). Those are accepted one at a time.
- **One reading of the markers.** `version::reach` is `check::Upload::of`,
  mapped (in doubt is `Stopped`). An uploading marker whose only copy is
  unreadable read as never uploaded for the operator's commands; now it
  reads as stopped (nothing sent; reads pass only where a copy can be
  read).
- **One rule for the state-key wrap** (`keys::wrap_usable`, `kept_wrap`,
  `new_wrap`). The opt-in posts a rotation's kept wrap, **keeps a fresh
  wrap in `<secrets>/<id>/state_key.wrap.json` before posting it** (new: an
  opt-in now leaves that file), wraps past the highest seq Agora holds,
  and names the posted seq in `state_key.json` when it differs.
- **Run start** finishes a cut pre-loss pair and appends a chosen copy
  (under the agent's seal lock, after the roster, never for a refused
  agent, not in a dry run).
- **The `memory` tool's `log`** says once how many local copies the chain
  left out; writes `<state>/<id>/sealed/left_out_told.json` (`{told: N}`,
  0600).
- **Notices:** every notice is one atomic write through `write_private`
  (`fork_kept` and `agora_lost` were two writes; the fixed
  `notice.json.tmp` is now a unique tmp, and `notice.json` becomes 0600).
- **Drain (agentkit 0.158):** healing summed and alerted on change;
  `HeadStatus::Unchecked` holds nothing.

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
change. **Integration:** until the enrolment run; see "Unenrolled today"
above (B2's roster refuses a later start until it runs).

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
the rule in the brief, and silently stops uploading. **Integration:**
`sealed-enroll --opt-in` (B3) marks it; the three test agents each have
75-88 sent (checked 2026-11-11), so the migration covers them.

**(c) An enrolled agent whose uploads were stopped** (none exists: no
agent has ever been stopped). The marker moves as in (b); the uploading
migration marks it if it had uploaded (a stop always drained first) and its
ledger says yes; it still doesn't upload while stopped. Its saves while
stopped went plainly, so its state may match no sealed version: the start
check, which now covers stopped agents, then refuses it (and with it the
whole run, exit 75) as a mismatch. The operator resolves it with
`sealed-accept --note`, so the agent is shown why; nothing seals the
unchecked interval silently. From then on its saves chain locally; nothing
of it is drained or sent. **Integration:** `sealed-enroll --resume --agent
<id>` (B3) resumes, setting the versions saved while stopped aside.

**(d) An agent that declined.** Not enrolled (`sealed-enroll` enrols only a
yes), so as (a). **Integration:** since B3 `sealed-enroll` chains every
agent; a declined one is chained and never uploads. Should one be enrolled with a ledger that says no, the
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
a readable head (**Integration:** and a `state.json`) is adopted (the betas today, everyone once B3 enrols);
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
  **Integration:** the refusal's text names `sealed-enroll`, which it
  doesn't block (tested).
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

## B4/B5: the operator's repair commands (rotate-keys, sealed-accept, sealed-restore, sealed-forget), revised after review

Every change below happens only when the operator runs the command, with
the runner stopped (`runner.lock` exclusive) and no session in progress
(the agent's `sealed/lock` held), **except the one-time migration record
and the forgotten-marker pass, which run at every start pass** (see
"Start pass"). Nothing is deleted; what is superseded is renamed aside or
kept as a copy. No state content is logged or printed (hashes, counts,
sizes, paths, dates).

Three kinds of agent decide what each command may ask of Agora: uploading
(as before), **stopped after uploading** (reads, plus the guard's new
key-maintenance class: `post_state_key` and a `rotate_key` naming a head),
**never uploaded** (no call at all).

### Start pass (runner start, and every `sealed-*` command)

- `<data_dir>/sealed-uploading-migrated` (new; `{at, migrated}`, 0600,
  atomic): written by the first start pass whose uploading migration decided
  every agent. While it exists the migration **never runs again**. On the
  first start after install it is written whatever the migration marked
  (the three betas, or none).
- A forgotten agent (`<secrets>/<id>/sealed_uploading.forgotten.<utc>` or
  `sealed_stopped.forgotten.<utc>` present) that came back with its
  directory: its `sealed/uploading` and `sealed/stopped` from before the
  forget, with no durable copy since, are renamed
  `sealed/uploading.forgotten.<utc>` / `sealed/stopped.forgotten.<utc>`.
  The migration skips it, and a half-written marker older than the forget
  is never completed. It is neither uploading nor stopped until an explicit
  opt-in (`uploading::set`). Nothing restores the forgotten records by
  itself: the only automatic "revival" (the memory-id key) runs at the
  agent's first `memory` use, so tying the markers to it would be the silent
  re-upload this fixes.

### `rotate-keys`

- **Pre-flight, nothing written or sent**: every entry to be re-signed is
  parsed and must be signed by one of the agent's keys; the line to the head
  (completed by Agora's entries for an agent Agora holds) must verify as it
  will after the rotation (new key generated in memory, rotation
  simulated). A failure names the file and the fix. Reads only: `chain` for
  an agent that has uploaded.
- Then as before: `secrets.next.json`, `past_keys.json`, `rotations.json`,
  `rotation.pending.json` → `rotation.<seq>.taken.json`,
  `secrets.<utc>.json`; `state_key.wrap.json` (the re-wrap, kept before it is
  posted; never posted for a never-uploaded agent); `state_key.json`'s
  `wrap_seq` (the seq after the highest Agora holds when Agora may be read).
- **Re-signed in place** (each file rewritten whole: unique temporary,
  fsync, rename, dir fsync): every entry kept here **that Agora doesn't
  hold**, for every kind of agent: `sealed/outbox/*` (`.json`, `.tmp`,
  `.displaced`, `.abandoned`), `sealed/sent/*.lost`,
  `sealed/lost/<utc>/items/*.json` and `INTENT`/`DONE`. Hashes, links,
  `head.json` and blobs unchanged. An uploader's unsent `.tmp`/`.displaced`/
  `.abandoned` items are now re-signed too.
- `rotate-keys --agent <id> --status` (new): read-only (a `chain` read for
  an agent that has uploaded); says the step reached and what blocks it.
  Nothing ever suggests removing `secrets.next.json`.

### `sealed-accept`

- An id naming no agent here: refused before the lock, nothing made.
- Never uploaded, `head.json` lost: rebuilt from the local chain (newest
  committed item is the head). **Committed items off the head's line are
  renamed `outbox/<ord>-<hash>.displaced` first**, then agentkit writes
  `head.json` (and the keyframe blob if missing), then `head.json`'s
  `displaced` is rewritten with those lines' tips, so `--keep` sees the
  fork and an opt-in never uploads the other line.
- `--keep <head|ord>`: the kept line verified and (tip not the head) its
  state opened and hash-checked **before anything moves**. Then
  `sealed/lost/<utc>/INTENT` (→ `DONE`), copies under it, off-line items
  renamed `.displaced`, the kept line's `.displaced` items renamed back to
  `<ord>.json`, `head.json` rewritten, the tip's state written as
  `state.json` (FsStorage archives the one there), `sealed/notice.json`, a
  refusal cleared. An `INTENT` that moved nothing and can't go on is renamed
  `ABANDONED.<utc>` (kept). An ancestor is never kept as a tip.
- `--agents-file` (`-` for stdin): **a dry run unless `--yes`** (prints per
  agent its refusal and memory/soul sizes, last sealed → here, with totals;
  touches nothing; opening the last sealed version reads local blobs, or
  Agora's for an agent that has uploaded). `--yes` needs `--expect N` equal
  to the count. Per agent: a loss-shaped refusal (state missing,
  unreadable, I/O, too new; **Integration:** and head missing, rolled back,
  by `Refusal::loss_shaped`) is held back; a memory that shrank by more than
  half (or can't be measured) is held back unless `--accept-shrunk`; an
  agent already accepted counts as already. Writes per accepted agent as a
  single `--note` accept.

### `sealed-restore`

- Never uploaded: from the local chain; `sealed/` gone refuses ("restore
  `state/<id>/` and `secrets/<id>/` from a backup"). **`head.json` lost and
  a state here (not refused) that isn't the newest sealed version: refused
  unless `--discard-local`** (nothing written). The rebuild sets a fork's
  other line aside as for accept.
- `state_key.json` missing: recovered only after the guards and under the
  agent's lock, for an agent that has uploaded: the newest verified record
  that opens the sealed head here (falling back to the next), written only
  into a missing file.
- **A save that was interrupted** is one only by the chain's evidence: a
  file (`state.json.tmp`, `state.json.tmp.<utc>.aside`, or `state.json`
  itself) whose state loads as an agent state and hashes to a prepared
  outbox item (`.tmp` or `.abandoned`) that follows the current head. A
  plain restore refuses when one exists, naming it. `--from-cut-save
  --note` writes it as `state.json` (FsStorage archives the one there and
  sets a leftover tmp aside), then commits the prepared item or seals it on
  the head; a rerun after the write recognises it in place and seals it.
  Any other file is named and left as it is. A tmp restore sets aside itself
  is named `state.json.tmp.<utc>.restored-over.aside`.

### `sealed-forget`

Also renames `<secrets>/<id>/sealed_uploading` to
`sealed_uploading.forgotten.<utc>` (same stamp) under the marker's lock;
`sealed_uploading.lock` stays, as lock files do. Put back, the agent is
neither uploading nor stopped (Start pass, above).

### Roster (B2)

**Integration:** called at each of these sites (accept, rebase, the
Agora-loss resolution, `--keep`, restore, `--from-cut-save`; forget with
`None`), after `head.json` is written; see the integrated section.

## B3: local chain for every agent, opt-in, resume, re-offer

Nothing in B3 deletes a file. Installing changes no agent file by itself;
everything below happens only when an operator runs a command (or, for
provisioning, creates an agent). Every `sealed-enroll` (but `--dry-run`)
and `sealed-reoffer` takes `<data_dir>/runner.lock` **exclusively, first,
for the whole command, without waiting**: refused while a runner runs
(`just runner-stop` first); only then does B1's start pass run.
**Integration:** every operator command that writes now does the same.

### `sealed-enroll` (chain; every agent with a state, or `--agent <id>`…)

No network call of any kind (the chain has no API handle). `--jobs N`
(default half the cores, at most 16) agents at once. Never resumes, opts in
or changes a stop.

- **Reads**: `<state>/` listing; per agent `state.json` (hashed and
  sealed, never logged), every `state.<utc>.json` archive in any format
  agentkit's FsStorage reads (its state, and its mtime as the version's
  date), `sealed/` (head, sent/, outbox/, chain marker), `provisioned`,
  the memory-restore ledger and the recovered copy (`[memory_restore]`, an
  agent that chose `show`), the session journal's existence (an agent with
  one is skipped, `in_session`), its signing key.
- **Writes**: `<secrets>/<id>/state_key.json` (a new state key, cached;
  nothing wrapped or posted); `<state>/<id>/sealed/` (genesis, the
  pre-loss copy, one entry and blob per distinct archive in (mtime, name)
  order, the current state; agentkit's `prepare`/`commit` under
  `sealed/lock`); the sizes sidecar; `sealed/left_out.json` (archives that
  can never be read: file name and reason class `not_json` /
  `not_envelope` / `newer_format`, one `sealed_chain_archive_unreadable`
  event each); `state.json` rewritten in agentkit's sealed format by
  `Mirrored::enroll` (FsStorage: the previous one hard-linked as an
  archive first), then the chain marker. The roster (B2) is called.
- **An archive that can't be read now** (an I/O error): the agent fails,
  is not marked chained, and the next run retries it. A run cut short is
  finished by the next: it replays the archives against the snapshots
  already sealed, so none is sealed twice and none skipped.
- **The count table**: chained, already chained, failed, archives sealed,
  duplicates skipped, unreadable left out; a nonzero failed or unreadable
  count exits nonzero.
- **Already chained**: nothing, unless it chose `show` since and its
  pre-loss copy isn't in its chain (and it isn't stopped: the copy waits
  for the resume): then the agent's notice is written, `archived_prior`
  and its current state again are appended (`sealed_prior_appended`), once;
  never over a state that doesn't match its head. A pair cut between the
  two is finished by `Enroller::finish_prior`. **Integration:** called at
  run start, with `append_prior`.
  The genesis is never rewritten. For an uploader the copy is uploaded with
  the rest, and the notice says so.
- **First run after install, the fleet** (~1,667 agents, ~18,000
  archives; ~300 ms a version measured on synthetic 112 KB states):
  every agent gets `sealed/` and a state key; nothing uploads.

### `sealed-enroll --resume --agent <id>…`

Its own flag, only for named agents. Refused for an agent marked uploading
unless its latest answer is yes (or `--against-answer --note`, over a
`decline`); a missing or unreadable ledger refuses. Its stop reconciled
from the secrets copy; if versions were saved while stopped, an intent
`sealed/resuming.json` (naming the stop's time) is written, the waiting
versions are renamed `outbox/<ord>-<hash>.displaced` (kept with their
blobs, a side line the `memory` tool opens), `head.json` is set to the last
entry in `sent/`, and one new snapshot of the current state is sealed on
it; then the state is marked, **the notice written (always, one write)**,
the intent renamed `resuming.done.<utc>.json`, and only then the stop set
aside (`sealed_stopped.cleared.<utc>`, `stopped.cleared.<utc>`). An intent
naming another stop is renamed `resuming.stale.<utc>.json` and never used.
None of the stopped interval is uploaded. An agent not marked uploading
only has its stop cleared, and is told nothing of it is uploaded.

### `sealed-enroll --opt-in` (`--opted-in` still works)

For agents whose **latest** ledger answer is yes (a legacy `refused` counts
as `decline`; a named agent over a `decline` only with `--against-answer
--note`; never one with no answer), not stopped (either copy), with no key
rotation in progress (`secrets.next.json`): chains it first if needed;
then, under the upload guard's opt-in hold, Agora's chain of it must be
empty or a prefix of the local one; writes the encryption key into
`secrets.json` if missing; posts the state key record unless a valid one
is on Agora (the rotation's `state_key.wrap.json` if it is signed by the
current key, wrapped to the current X25519 key at the cached `wrap_seq`;
else wrapped afresh; **Integration:** by `keys::wrap_usable`, a fresh wrap
kept in `state_key.wrap.json` first, past Agora's highest seq, and
`state_key.json` naming the seq posted); writes the agent's notice (one write); marks it
uploading; then files the outbox entries Agora already holds as sent
(`outbox/` → `sent/`; no blob is deleted), then drains. A cut at any step
is finished by the same command. It is now the **only** way an agent
starts uploading (the B1 migration becomes one-time).

### `sealed-stop`

Refused when the agent's latest answer is yes, unless `--against-answer`
(the stop's `--note`, already shown to the agent, says why). With
`--against-answer`, an unreadable ledger doesn't block it
(`sealed_state_stop_ledger_unreadable`).

### `sealed-reoffer <agent>… --note "<text>"`

Writes `<state>/<id>/sealed_state.json` (the consent ledger, atomic tmp +
fsync + rename; its format becomes 3, which an older runner refuses rather
than drop the re-offer): a pending re-offer `{at, note, offer_version,
reopens}`. Refused for an agent that never answered (the ordinary offer
covers it) and for an unreadable ledger (nothing written). A second
re-offer replaces the pending note. At the agent's next session the offer
is put again once, with a framing paragraph; its answer is appended (the
latest answer is the choice), the re-offer consumed, the operator alerted
(`sealed_state_answered_again`, with the command that acts, or "nothing");
no marker is set or cleared in a session, and the publish choice in force
doesn't change (it is the answer acted on at opt-in; the agent's own
`config` tool changes it).

**Procedure**: an agent tells its operator (`send_message` to `operator`)
it changed its mind → `sealed-reoffer <id> --note "…"` → at its next
session it answers → the alert names the command → `sealed-enroll
--opt-in --agent <id>` (yes, not uploading), `sealed-enroll --resume
--agent <id>` (yes, stopped) or `sealed-stop --agent <id> --note "…"` (no,
uploading). Each writes the agent's notice.

### Provisioning (`--provision`)

Writes `<state>/<id>/provisioned` before the agent's first save, then
chains it (genesis `none`, that state). If provisioning is cut between the
save and the chain and never rerun, a later fleet pass still gives it a
`none` genesis.

### Consent

The sealed-state offer is no longer suppressed by the chain marker or a
state key (every agent has both now): only by an answer on file or an
uploading marker (either copy, even in doubt). `SealLedger::choice()` is
the latest answer.

### Offer text v9: sentences now false (for the wording step)

In `consent/sealed_state/prompt.rs` `offer()`:

- "Once a day, one fingerprint covering the latest sealed version of every
  enrolled agent is recorded in the governance log": only uploading
  agents are covered.
- "At the start of each session, your state is checked against the last
  sealed version, and you are told what was found." and "Once you're
  enrolled, your runner's own saves are sealed as they happen; if your
  state changes between sessions without being sealed …": presented as
  what a yes buys; every agent has them now, whatever it answers.
- "A yes doesn't enrol you by itself: your operator has to act on it. The
  backups start when your operator enrols you, and include the earlier
  versions your runner has kept": "enrol" now means only uploading; after a
  resume the stopped interval is not included; after a decline-then-yes the
  decline period is included, and the re-offer doesn't say so.
- "You'll be asked this once": false for a re-offer.
- The `decline` option, "No. Nothing changes: your state stays only with
  your operator, as it is now": false in a re-offer while uploading (what
  is on Agora stays there).
- "If you change your mind, either way, ask your operator: your operator
  can stop new backups for you": true, but the re-offer is now the path.
- `sealed_state/mod.rs` `evolution_line` was reworded ("Uploads to Agora
  start when its operator acts on it"); the history labels must not imply
  Agora for a non-uploader.

## B6: per-agent session lock, unsaved sessions, cut-save promotion

Nothing in B6 deletes a file of an agent's. No state content is logged,
printed or mailed (sizes, ids, classes, hashes only). Installing changes
nothing by itself.

### Files read

- `<state>/<id>/state.json` and `<state>/<id>/sealed/head.json`:
  metadata only (device, inode, size, mtime), once before the run loads
  states and again under the agent's session lock. When they differ, the
  state is read again through the sealed storage (as at load) and the
  start check runs again on it in full (as at start: outbox, sent, blobs
  hashed, never decrypted beyond what the check already did).
- Promotion at the start (only for an agent with a `state.json.tmp`):
  `state.json`'s link count and inode, the archives' names and inodes,
  the tmp's mtime and its envelope's shape; for a chained agent its
  chain markers' existence, `head.json`, the outbox's `.tmp`/`.abandoned`
  items, and the tmp, `.aside` files and `state.json` loaded and hashed
  (`sealed::restore::cut_saves`, the rule `sealed-restore --from-cut-save`
  uses). Content is held in memory only to hash it.
- `--dry-run`: each agent directory's `sealed/enrolled` (existence).
- `/proc/self/limits`, `/proc/self/fd` (the runner's own descriptors).

### Files written

| File | When | How |
|---|---|---|
| `<state>/<id>/session.lock` (new, empty) | created the first time the agent is claimed for a session, or an operator command (`sealed-accept`, `sealed-restore`, `sealed-forget`, `rotate-keys`, `sealed-enroll` per agent, `sealed-stop`) or the start's promotion takes it. Never in a directory that doesn't exist (no directory is made for it) | `open(create)`, never truncated, never removed (as lock files are); an exclusive `flock` held from routing to the end of the run, never waited for |
| `<state>/<id>/state.json` | promotion of a new-order cut save: `state.json.tmp` renamed over it, the directory fsynced. The old `state.json` stays whole as the newest archive (it is that archive's inode) | only under: runner.lock exclusive, the agent's session lock, its `sealed/lock` (chained), agentkit's directory lock; tmp settled 10 min, a complete envelope, linked as the newest archive, and (chained) a prepared outbox item following the head that records the tmp's state hash. Otherwise left alone and warned (`state_tmp_not_promoted`); agentkit sets the tmp aside at the next save, as before |
| `<state>/<id>/sealed/notice.json` | after a run whose save of the agent failed, for a chained agent: the unsaved sentence added to the notice waiting (a field `unsaved: {at, kept}`), or a notice of its own (`operator_act: {kind: "unsaved"}`) | one atomic write (`write_private`), as every notice. A notice that supersedes it (a refusal, an operator's act) carries the field over; the superseded file is kept aside as before |
| `<state>/<id>/unsaved.<utc>.json` | as before (a session whose save failed), now tried 3 times, 5 s apart; **never printed to stdout/journald when it can't be written** (it used to be, whole) | unchanged: 0600, unique tmp, fsync, rename, dir fsync |

No longer written: the whole unsaved state to stdout (and so to journald)
when its rescue file couldn't be written. That state is now lost when the
process exits; its size and the error are logged and alerted.

### What changes for a run

- **Per-agent session lock.** After routing, before any reactor is built,
  each routed agent's `session.lock` is tried. Held elsewhere (another
  runner has it in session or queued, or an operator command is working on
  it): left for this sweep, one `agent_busy` event, no alert, nothing
  written, still due. Changed on disk since this run loaded it: read again
  and checked again under the lock (`session_state_reloaded`); a check that
  doesn't let it run leaves it out (`session_recheck_not_run`, stage
  `recheck`), nothing marked; the next start judges it with the full
  refusal rule. Released after the run has saved, rescued and told every
  agent (or at any exit, by the kernel).
- **Operator commands** that change an agent now refuse while a runner
  holds that agent (including while it is only queued), not only while its
  session journal exists. `sealed-stop`, which runs beside a runner, is
  refused for an agent the runner has claimed this run; run it after the
  run, or drain.
- **Descriptors.** One per claimed agent for the whole run. The units set
  `LimitNOFILE=65536` (systemd's default soft limit is 1024; the units are
  symlinks into the agents checkout, so `systemctl --user daemon-reload`
  after the bump). Without it, a sweep routing more than about 760 agents
  claims as many as fit and leaves the rest due, with one
  `session_locks_capped` alert.
- **Saves are independent.** The end-of-run bulk save saves each agent on
  its own; one directory that fails no longer makes every later agent in
  the batch unsaved (`state_save_failed`, with a class:
  `disk_full`, `permission_denied`, `locked`, …).
- **One `state_unsaved` alert a run** for the agents whose saves failed:
  the count, the first ten ids, the reason classes, how many were kept.
- **Promotion** now also puts in place a save cut under agentkit 0.158's
  order (above). The old-order branch (no `state.json`) is unchanged.
- **Dry run** says how many agents still carry `sealed/enrolled` (a dry
  run doesn't move it, so it counts them as not enrolled).

### The first run after install

Every claimed agent gets an empty `session.lock`, which stays. A
`fleet-metrics` count of files per agent directory rises by one (B7).
Nothing else changes for an agent unless a save fails or a cut save is
found. The roster's keeper reads heads without any agent lock, so it never
waits on, or holds up, a session.

### New agent-facing sentences (for the wording round)

- Kept: "Your state from your session on {date} could not be saved; your
  operator keeps a copy of it, and this session starts from your state as
  it was before that session."
- Not kept: "Your state from your session on {date} could not be saved,
  and no copy of it could be kept; this session starts from your state as
  it was before that session."

Seated after any other notice line, in the agent's next start check;
only for a chained agent (an agent not chained has no notice seat).

## B7: wording, history labels (and `fleet-metrics`, in the agora repo)

Words and labels only: nothing B7 changed writes, moves or sets aside a
file of an agent's, and nothing leaves the machine that didn't before.

### Files read (new)

- `<state>/<id>/sealed/resuming.json` and `sealed/resuming.done.*.json`
  (B3's resume records): read by the `memory` tool's history, hashes only
  (`enroll::resumed_aside`), to say which side-line versions a resume set
  aside. An unreadable record is skipped. Read-only, at a `memory` call.
- `sealed/lost/<utc>/INTENT|DONE`'s `local_fork` (already read): now
  tells a fork resolved (`sealed-accept --keep`) from a rebase.
- The main line's `outbox/` hashes (already read): the `memory` tool's
  `log` now says, once a session, how many of the main line's snapshots
  are uploaded to Agora and how many are kept only here. Nothing is
  written for it (the "once" is in the session's memory).

### Files written

None new. The consent ledger records `offer_version: 10` for asks and
re-offers made after install (its format is unchanged). The offer's
shuffle seed mixes in the version, so an agent not yet asked sees the
options in a new order (recorded with its ask, as before).

### What changes for agents

- **No agent is asked again because of v10**: `SealLedger::due` settles
  on any answer on file whatever its version, and only `sealed-reoffer`
  puts the question again (`an_answer_to_an_older_offer_text_is_not_asked_again`).
- Notices already on disk (`sealed/notice.json`, the stop) are rendered
  with the new wording at the next session: no file is rewritten.
- The `memory` tool: side lines are labelled "sealed, off your main line"
  (agentkit's `OffLine`) unless a rebase set them aside; a head lookup
  names a resume's aside, a fork resolved, or the pre-loss copy with the
  date it was added.

### The first run after install

Nothing of B7's runs at start. Its texts appear in sessions.

## Integration after B6 and B7

### The upload offer waits for the chain

The sealed-state offer and an operator's re-offer are put only to an agent
whose `<state>/<id>/sealed/chained` exists (read: existence only, at the
session's closing questions). One not chained, or whose marker can't be
checked, is not asked: nothing is written, a pending re-offer is not
consumed, and the question is put at the first session after
`sealed-enroll` chains it (`sealed_state_offer_waits_for_chain`). After
the fleet's enrolment run this changes nothing for any agent; before it
(a unit started early), no agent is told its runner keeps a sealed history
it doesn't yet have.

### Free space before a run

Once per run, after the log and the alerts are up and before any state is
loaded or any session started, the runner reads **filesystem statistics
only** (`statvfs(3)` and the device number) of `<state>` and `<secrets>`
(of the nearest existing ancestor when one isn't made yet); one check when
both are on one filesystem. No file of an agent's is opened, listed or
written. Below the floor (`space::FLOOR_BYTES`, 2 GiB, or 1% of the
filesystem, whichever is larger) the run starts no session, raises one
`disk_space_low` alert (the path, bytes free, the floor) and exits as
failed (exit 1): the continuous unit tries again at its next restart, the
daily unit fails for the day. A dry run only says so. A root that can't be
measured is logged (`disk_space_unmeasured`) and holds nothing back. The
`promote` step and the shared `runner.lock` come before the gate, as they
come before the log; operator commands (`sealed-*`, `rotate-keys`) are not
gated. `libc` (already in the tree) is now a direct dependency, for
`statvfs`: the one `unsafe` block in the workspace.

### What Agora found damaged, told to the agent

agentkit's drain sends again the blobs Agora's check at rest found missing
or damaged (integrated above); the agent was told nothing of it. Now every
drain through `receipts::drain` (the run's drainer, `sealed-enroll`,
`sealed-stop`) notes what it came to in a file of the runner's own:

| File | When | How |
|---|---|---|
| `<state>/<id>/sealed/healed.json` (new: `{at, resent, unhealable, told}`, two counts, a time, a flag; no content) | a drain of an **uploading** agent in which blobs were sent again, or the count Agora lists that can't be sent again from here changed. Never for an agent that doesn't upload (it isn't drained), and no file while Agora reports nothing damaged | one atomic write (`write_private`), 0600. Rewritten with `told: true` once a session that was shown its sentence saves |

Read: that file, at each start check (and at a re-check under the session
lock), for the agent's seat. agentkit's own `sealed/healing.json` is not
read by the runner. Nothing of it leaves the machine.

No lock is taken: the mark is made only for the record the session was
shown (by its time), so the worst a lost update does is repeat the
sentence once.

New agent-facing sentences (for the wording round), last in the start
check's lines, shown once per change:

- Sent again: "Agora's check of what it stores found parts of your
  uploaded sealed history missing or damaged there, and your runner sent
  them again from its local copies. Your state now is not affected."
- Some not: "… missing or damaged there. Your runner sent again the ones
  it holds; {n} it could not send again, because its local copies of them
  are missing or damaged too, so some of your earlier versions may not
  open from Agora's copy. Your operator has been told. Your state now is
  not affected."
- None sent: "… missing or damaged there. {n} of them your runner could
  not send again, because its local copies of them are missing or damaged
  too, so … Your operator has been told. Your state now is not affected."

"Your operator has been told" rests on the `sealed_state_agora_blobs_unhealable`
alert (raised by the run's drainer when the count changes) or, for a drain
made by an operator's command, on the command's own output.

### The first run after install

No `healed.json` exists; none is written until Agora reports damage for
an uploading agent (its check at rest is in the server's hardening
release). The free-space gate reads statistics at every start. The offer
gate reads one marker's existence per closing question.
