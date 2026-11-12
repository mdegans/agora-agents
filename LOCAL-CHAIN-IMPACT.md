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
`sealed-stop` runs beside a runner, and so does `sealed-enroll --opt-in`
without `--yes` (a dry run: no start pass, no lock; review round
2026-11-12).

**Data-directory files, once:** (`<data_dir>/sealed-uploading-migrated`
is no longer written: the one-time migration is tied to the old chain
marker, see the CONSENT review round below), `<data_dir>/sealed-roster.json`,
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

*First start.* While it still has `sealed/enrolled` (the old chain
marker), the one-time uploading migration finds each with `sent/`
non-empty and a yes in force in its ledger: writes
`<secrets>/<id>/sealed_uploading.setting` (the intent), `sealed/uploading`,
then renames the intent to `<secrets>/<id>/sealed_uploading` (and its
`.lock`). Only then `sealed/enrolled` → `sealed/chained` (agentkit, mtime
and bytes kept), after which the migration can never judge that agent
again (review round 2026-11-12). Adopted into the roster. The full start check against
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
   start pass itself (marker move, migration) and begins the roster with
   the first agent it chains. **Review round, 2026-11-12:** if the data
   directory already holds chained agents and no roster (agents chained
   by an older runner, as the test agents' own data directories are), it
   refuses before chaining anything and names `agora-seed sealed-enroll
   --begin-roster --note "<why>"`: run that once first, then this. It
   exits nonzero while any agent with a state is left without a chain,
   naming each by id and why (see "Review round, 2026-11-12: STATE").
   Estimated 7-12 minutes (synthetic: about 0.3 s a version).
7. Fleet metrics again; start the units; compare after the first sweep.

If a unit starts before step 6: with no chained agent in the data
directory the first start runs (beginning an empty roster and alerting the
unchained count) and every later start refuses, naming `sealed-enroll`,
which that refusal doesn't block. With chained agents and no roster the
start refuses at once, naming `--begin-roster` (review round,
2026-11-12). Integration tests both sequences.

## Upgrading a data directory that already has uploading agents

For a data directory where an older runner chained and uploaded agents
(agentkit's old chain marker `sealed/enrolled`, a non-empty `sealed/sent/`,
a yes on file in the ledger, no uploading marker, and no
`sealed-roster.json`): the three test agents are in exactly this state.
Integration (2026-11-12) checked the order end to end on synthetic state
(`review_tests::an_agent_uploading_under_the_old_build_upgrades_by_the_documented_steps`);
nothing in the order was changed.

What happens, in order, whichever command runs first:

- The start pass (every runner start and every non-dry `sealed-*`
  command, `--begin-roster` included) judges each agent with the old
  marker *while the marker is there* (`uploading::migrate`): `sent/`
  non-empty and an answer in force of yes, with no re-offer waiting →
  marked uploading (`UploadingBy::Migration`). Then agentkit's
  `migrate_marker` moves `enrolled` to `chained`. Once the old marker is
  gone the agent is never judged again: one that didn't qualify (no yes
  in force, a re-offer waiting, an unreadable ledger) stays not
  uploading until `sealed-enroll --opt-in` on a yes. It fails toward not
  uploading, never toward stuck.
- A migration that failed on I/O keeps the old marker and is tried at the
  next start; meanwhile `sealed-enroll` (chain) refuses that agent rather
  than chain it a second time.
- With chained directories and no roster, the runner's start refuses the
  whole run and plain `sealed-enroll` refuses before chaining anything;
  both name `--begin-roster`. Nothing is marked refused for it.

The operator's steps (the runner stopped throughout, `just runner-stop`):

1. `agora-seed sealed-enroll --begin-roster --note "upgrade: agents
   uploading under the build before the roster"`. Its start pass marks and
   moves (above); then the roster is begun from the chained directories
   as they are, each adopted at its head; the note is kept in
   `sealed-roster.began`.
2. `agora-seed sealed-enroll --jobs 16`: chains every other agent with a
   state; the upgraded agents are already chained and are left as they
   are.
3. Optional: `agora-seed sealed-enroll --opt-in` (a dry run) lists any
   agent with a yes in force that is not uploading (for example one that
   had a re-offer waiting at the migration); `--yes --expect N` starts
   them.
4. Start the runner. Each upgraded agent is checked against Agora's copy
   and drained as an uploader: nothing in `sent/` is sent again, its head
   and state are unchanged.

If the runner is started before step 1, the start does the marking and
moving and then refuses; step 1 then works the same way.

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
  one alert. ~~Deleting the roster, `began` and the witness together reads as
  a first run.~~ **Review round, 2026-11-12:** closed while any chained
  directory remains: the start refuses (`NotBegun`). It still reads as a
  first run when every agent's chain marker went with them, where the
  state keys left in `secrets/` make three or more of them a refusal.
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

**Superseded, review round 2026-11-12:** a start no longer adopts chained
directories when there is no roster. See "Review round, 2026-11-12: STATE".

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

- (Superseded, review round 2026-11-12: no `sealed-uploading-migrated`
  record is written. The uploading migration judges an agent only while
  it still has the old chain marker `sealed/enrolled`, and the same pass
  then moves that marker, so it cannot run for an agent twice.)
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
  agent that chose `show`), the session journal's existence (**review
  round, 2026-11-12:** no longer a skip; the agent is chained from its
  last saved state and the journal is left untouched), its signing key.
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

Its own flag, only for named agents. Refused for an agent with an
uploading marker (sound or in doubt) unless its answer **in force** is yes
with no re-offer waiting; no flag overrides that (`--against-answer` is
gone here, review round 2026-11-12); a missing or unreadable ledger
refuses. A marker in doubt is replaced on that yes. Its stop reconciled
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

**A dry run without `--yes`** (the count and the ids; no start pass, no
write, no network); `--yes` acts, and with no `--agent` also needs
`--expect N`, the count the dry run gave (a mismatch touches nothing).
For agents whose answer **in force** is yes with no re-offer waiting (a
legacy `refused` counts as `decline`; an answer given with no question
open is never in force; nothing opts an agent in against its answer:
`--against-answer` is gone here), not stopped (either copy), with no key
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

Refused when the agent's answer in force is yes, or while a re-offer
waits for its answer, unless `--against-answer` (the stop's `--note`,
already shown to the agent, says why). With `--against-answer`, an
unreadable ledger doesn't block it (`sealed_state_stop_ledger_unreadable`).
Past that check the stop is unconditional and sends nothing: see the
CONSENT review round below.

### `sealed-reoffer <agent>… --note "<text>"`

Writes `<state>/<id>/sealed_state.json` (the consent ledger, atomic tmp +
fsync + rename; its format becomes 3, which an older runner refuses rather
than drop the re-offer): a pending re-offer `{at, note, offer_version,
reopens}`. Refused for an agent that never answered (the ordinary offer
covers it) and for an unreadable ledger (nothing written). A second
re-offer replaces the pending note. At the agent's next session the offer
is put again once (whatever `[sealed_state]` says: off, `ask = false`,
unlisted), with a framing paragraph; its answer is appended (it becomes
the answer in force), the re-offer consumed, the operator alerted
(`sealed_state_answered_again`, with the command that acts, or "nothing");
no marker is set or cleared in a session, and the publish choice in force
doesn't change (it is the answer acted on at opt-in; the agent's own
`config` tool changes it).

**Procedure**: an agent tells its operator (`send_message` to `operator`)
it changed its mind → `sealed-reoffer <id> --note "…"` → at its next
session it answers → the alert names the command → `sealed-enroll
--opt-in --agent <id> --yes` (yes, not uploading), `sealed-enroll --resume
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
uploading marker (either copy, even in doubt). `SealLedger::in_force()`
is the one reading every command acts on (review round 2026-11-12).

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

## Review round, 2026-11-12: STATE

Fixes for the independent review of state loss, stranding and wrongful
fleet stops. Synthetic states only in tests; nothing here was run against
live data.

### New file: `<data_dir>/sealed-unchainable.json`

Written **only** by `sealed-enroll` (not a dry run), under the exclusive
runner lock, at the end of a chain pass, atomically and 0600
(`write_private`); never created empty. Ids, a reason class and a date
per agent; nothing of any state. Classes: `no_signing_key`,
`state_unloadable`, `prior_copy_unreadable`, `state_key_lost`. An agent
enters it when enrolment finds it can't chain it for one of those, and
leaves it when a later enrolment chains it (or finds no state); one that
was merely not reached (`busy`, `failed`) neither enters nor leaves.

**Read at every start.** Agents on it that are still unchained are left
out of the run (they never run unchained), with one alert a run
(`sealed_state_unchained_agents`, details `left_out*`) and an
`agent_skipped` event each (stage `unchainable`), and are **not** counted
by the strays rule. Every other unchained state counts as before: three
or more refuse the run. An unreadable record excuses nobody.

Why: before, an agent enrolment could never chain was a stray at every
start, and three of them parked the fleet behind agents no rerun could
fix. What it costs: an agent on the record doesn't run until the operator
repairs it. That is deliberate (the alternative is running it with no
check), and the alert repeats every run.

### `sealed-enroll` (chain)

- An agent with a **session journal** is chained from its last saved
  state. The journal is not read, changed or removed: the agent's next
  session still shows its interrupted-session notice. No session can be
  live: the command holds `runner.lock` exclusively and takes each
  agent's session lock.
- Nothing is written for an agent it can't chain: the signing key, the
  state's loading and the pre-loss copy are all read before the state key
  is made or the genesis sealed.
- **No new state key over a chain already begun** (a `head.json` with no
  `state_key.json`): refused, class `state_key_lost`.
- An older runner's marker (`sealed/enrolled`) that couldn't be moved:
  the agent is not chained again.
- Exits nonzero while any agent with a state is left unchained, printing
  and logging (`sealed_enroll_unchained`) the count, the classes and the
  ids, and saying which count at the start. `--dry-run` reports the same
  classes and writes nothing (not the record either).
- `--resume` sets the roster to the resumed head (`roster::replace`), which
  may be below the head the roster saw while stopped.

### The roster: no silent begin, and the ways out

A roster is begun only (a) by the first agent enrolment chains in a data
directory with no other chained agent, (b) by a start that finds no
chained directory at all, or (c) by the operator:

    agora-seed sealed-enroll --begin-roster --note "<why>"

(runner stopped). It adopts every directory that is chained and has a
readable head and a `state.json`, as found; renames aside (never deletes)
the roster and witness it found (`sealed-roster.<was>.<utc>.json`,
`secrets/sealed-roster.witness.<was>.<utc>.json`); appends the note to
`sealed-roster.began`; logs `sealed_roster_begun`. It refuses a roster
that reads and matches its witness, and one in a newer format. What it
gives up: every head is taken as found, so a rollback made before it is
no longer told apart by the roster.

| The start says | What it means | Way out |
|---|---|---|
| `NotBegun` | no roster and no trace of one, but chained directories | the first install over agents an older runner chained, or everything at the data directory's top level lost: put the roster back from a backup, or `--begin-roster --note` |
| `Missing` | no roster, but `began` or the witness says there was one | put it back from a backup, or `--begin-roster --note` |
| `Shrunk` | the roster holds fewer agents than the witness says | put the newer roster back from a backup, or `--begin-roster --note` (both files set aside). Not by deleting the witness: it is the only record of the count |
| `Unreadable` | the roster or the witness can't be read or parsed | permissions and the disk first (the next start reads it again); then a backup, or `--begin-roster --note` (the damaged file set aside) |
| `Newer` | written by a newer runner | upgrade the runner; `--begin-roster` refuses |

The refusal's own text carries the same way out (`RosterError::way_out`).

New field: a roster entry may carry `pending: true`, set by
`sealed-accept` before it clears a refusal and cleared when it records
the head. An accept cut in between is finished by running it again.
Since the FINAL round it carries `pending_at` too (the lowest seq the
accept will set): a head at or past it recorded by a later run ends the
pending, and the rerun refuses a head below it.

### The start

- A refusal for sealed history that **can't be read** (`seal_unreadable`)
  writes no `sealed/refused` and no notice: the agent is left out of that
  run and judged afresh at the next. (A marker of that reason written by
  an earlier build of this branch still needs `sealed-accept`.)
- When one of the runner's own records fails (roster, secrets tree), the
  run is refused and **no agent is checked or marked**.
- An agent read again under its session lock (changed on disk since the
  load) goes through the start's own check: a refusal is marked, told
  and alerted; a chain marker gone since the load is a refusal
  (`head_missing`), never a session saved outside the chain; a session it
  had elsewhere counts for `min_cycle_secs`.
- A marker for a reason that isn't loss-shaped is **rewritten** with a
  loss found later (`sealed/refused`, atomically; the notice keeps its
  first date and reason).
- Unsaved sessions are rescued before the last drain and the roster's
  last pass; a reactor that ends with an error is alerted
  (`state_unsaved`, keyed by endpoint).
- Every routed agent busy in another process: exit 0,
  `all_routed_agents_busy`.
- A starting runner waits up to 20 s for `runner.lock` (another runner's
  cut-save promotion holds it exclusively for a moment).

### `sealed-restore`, `sealed-accept`, `rotate-keys`

- `sealed-restore` refuses a chain that was begun and never finished (a
  head, no chain marker, nothing on Agora, not refused, never on the
  roster), naming `sealed-enroll`: its head is an old archive's version.
- `sealed-restore` records the head in the roster **before** the chain
  marker and the refusal go, and no longer writes `sealed/published` for
  an agent that never uploaded (`write_private` where it does).
- `sealed-accept --keep` records the roster before marking the
  resolution done.
- `rotate-keys` refuses an agent whose chain was begun and never
  finished, naming `sealed-enroll`.

### Fleet metrics to compare in the dry run (counts only)

Besides the existing ones: agents on `sealed-unchainable.json` by class;
agents skipped at stage `unchainable`; `sealed_enroll_unchained` counts.
An agent on the record that had sessions before is a drop in sessions run
that the record explains; any other drop is not explained by this round.

## Review round, 2026-11-12: CONSENT

What changed in what leaves the machine, what is written, and what an
operator's command may do. Nothing here reads or prints agent state.

### What may start, resume or stop uploads

- **One reading**: `SealLedger::in_force()` (`consent/sealed_state/ledger.rs`)
  says which answer is in force and whether the operator may act on it
  now. Every caller uses it: the bulk list and the opt-in
  (`sealed/enroll.rs`), resume, `sealed-stop` without the override, the
  one-time uploading migration, the counts `sealed-enroll` prints.
- The answer in force is the agent's first answer, or the latest it gave
  to an operator's re-offer. **An answer given with no question open is
  kept on file and never in force** (it should be impossible; if it
  happens the agent is told at once that its earlier answer stands, no
  SOUL line is written for it, and the operator is alerted).
- **While a re-offer waits for its answer, nothing acts on the earlier
  one**: not an opt-in or a resume on its yes, not a stop on its no. A
  re-offer whose misses ran out still waits (re-open it again).
  `sealed-stop --against-answer --note` still stops.
- **The operator can no longer start or resume uploads against an answer
  that is not a yes.** `--against-answer` is removed from `sealed-enroll
  --opt-in` and `--resume`, with the notice variants that said "although
  your answer on file is `decline`". It stays for `sealed-stop`.
- `sealed-enroll --opt-in` is a **dry run** unless `--yes`; unnamed it
  also needs `--expect N`. The dry run takes no lock and runs no start
  pass.
- A pending re-offer is put, and its answer told back, **whatever
  `[sealed_state]` says** (off, `ask = false`, unlisted). The session now
  reads `sealed_state.json` for every agent, and keeps it (and may save
  it) only when the offer admits the agent or a re-offer is open. First
  offers stay the config's.

### `sealed-stop` is unconditional and sends nothing

Writes `<secrets>/<id>/sealed_stopped`, then `sealed/stopped`, first and
always, under the uploading marker's lock only. Not needed: the session
lock, a stopped runner, a reachable Agora, an agent whose state is in
order (refused, unsealed, changed, in a session, journal left: all
stopped the same). **No drain**: the versions that waited stay local and
the command prints how many. A runner mid-drain sends at most the item
already on the wire (agentkit asks `uploads` before each item). Stopping
twice is the same stop (its note and date kept); a copy that can't be
read is renamed `<name>.unreadable.<utc>` and rewritten; a missing
`sealed/` is never created (the secrets copy stands, the next start puts
`sealed/stopped` back). What couldn't be done now is printed with what
finishes it. Refused only for an agent with no uploading marker at all.
Once stopped, an uploading marker in doubt is settled from its readable
copy (`uploading::settle_stopped`; the other copy renamed
`<name>.doubt.<utc>`): the agent reads as "uploaded, stopped".

Consequence at a resume: everything in the outbox (not uploaded when the
stop began, or saved while stopped) is set aside as `.displaced` and
never sent. An agent opted in and stopped before anything of it reached
`sent/` can't be resumed by `--resume` (nothing known on Agora to
continue from): it stays stopped. **Known gap**, see the report.

### The uploading marker

- `set` writes an intent `<secrets>/<id>/sealed_uploading.setting`, then
  `sealed/uploading`, then **renames the intent** to
  `<secrets>/<id>/sealed_uploading`. A start finishes only that: local
  copy, no secrets copy, an intent saying the same. A secrets copy that
  was removed is never written back from the local one: the agent is in
  doubt, uploads nothing, and is alerted. So removing either copy (or
  both) stops uploads for good.
- A marker in doubt is ended only by a command acting on the answer in
  force: an opt-in or resume on a yes (`set_replacing_doubt`: both copies
  renamed `<name>.doubt.<utc>`, a new marker), or `sealed-stop`. The
  alert says so, and no longer says the agents "said yes" or to copy a
  marker by hand. Stopped agents are left out of the alert.
- `forgotten_at` also reads `state_key.forgotten.<utc>.json` and its
  note, so a forget of an agent that had no marker counts.
- **The one-time migration is tied to `sealed/enrolled`** and can't recur
  (no record file). Before installing, count (counts only) agents with a
  non-empty `sealed/sent/` and `sealed/chained` but no uploading marker:
  expected 0. Any such agent would not upload after install until
  `sealed-enroll --opt-in --agent <id> --yes` on its yes (the safe
  direction). An unreadable ledger at migration is an error event
  (`sealed_state_uploading_migration_ledger_unreadable`), not marked.

### What is sent for an agent that doesn't upload

- **Never uploaded: no `/api/state` request by any path.**
  `signer_registered` (a resumed signing rotation) now reads
  `POST /api/social/config` with nothing set, as `publish_chosen` does;
  a test drives every `SealedApi` method through the real client and a
  local server. Still sent for a never-uploader by `rotate-keys`: the
  signing-key rotation naming no entry (`/api/identity/rotate_key`), the new X25519
  public key (`/api/social/encryption_key`), that config read. Identity,
  nothing of its state.
- **Stopped** (uploaded before, marker sound): `rotate-keys` still reads
  its chain and state key records and sends a `rotate_key` naming
  Agora's head (the new public key, the head's hash and seq) and a
  re-wrapped state key record (the state key wrapped to the agent's own
  new X25519 key). No state. Kept so the uploaded history stays
  verifiable and openable. Each call is now logged
  (`sealed_state_call_while_not_uploading`: `call`, `write`, `what`; warn
  for writes, info for reads) and `rotate-keys` prints the list.
- **In doubt**: `rotate-keys` refuses before anything is written or sent,
  naming `sealed-stop` and the opt-in; the guard also refuses key
  maintenance for it (`UploadGate::may_maintain_keys`).

### Files and permissions

- Every consent ledger (`sealed_state.json` and its siblings) and every
  file written through `anchors::write_atomic` (`anchors.json`,
  `inclusion.json`, `receipts.json`, `published`, forget's note) is now
  written by `sealed::write_private`: 0600, a uniquely named temporary,
  fsync, rename, directory fsync. An existing wider file is replaced at
  its next write; nothing chmods the fleet in a pass.
- New names: `<secrets>/<id>/sealed_uploading.setting`,
  `sealed/uploading.doubt.<utc>`, `<secrets>/<id>/sealed_uploading.doubt.<utc>`,
  `sealed/stopped.unreadable.<utc>`, `<secrets>/<id>/sealed_stopped.unreadable.<utc>`.
  Nothing is deleted.

### Sentences changed (minimal; the TEXT round owns them)

- Stop note, first showing: "…the versions saved from now on stay here."
  → "…the versions not uploaded by then, and those saved from now on,
  stay here."
- Resume notice: "saved while uploads were stopped" → "not uploaded
  before the stop or saved while uploads were stopped" (one and many).
- Removed: both "although your answer on file is `decline`" variants.
- New, for an unprompted answer's receipt: "This question was not open,
  so this answer changes nothing: your earlier answer, `<choice>`,
  stands. To change it, ask your operator to re-open the question. You can
  tell your operator (the `send_message` tool, to `operator`)." (No claim
  that the operator was told: the alert is not awaited. Where what is in
  force is a legacy refusal, "your earlier answer, `<choice>`, stands" is
  "the earlier offer ended without an answer from you, which counts as
  `decline`, and that stands": see "Review round, 2026-11-12: FINAL".)

## Review round, 2026-11-12: TEXT

**What it is.** Wording, and the plumbing three sentences needed to be true. Offer v10 → v11
(`consent/sealed_state/prompt.rs`); every sentence an agent is told about its sealed history reworded
to be true in each state it can be shown in; one table of agent-facing text and one banned list
(`sealed/text.rs`) in place of five.

**Live state read, written, moved or deleted.** No new file, no deletion, no move, nothing leaves
the machine that didn't before.
- **Read, new:** the start check of an agent with no uploading marker lists its `sealed/sent/`
  (file names and entries, as the uploading path already did) to decide what its verdict says of
  uploads. No server call is added: such an agent is still asked nothing of Agora. Empty for an
  agent that never uploaded.
- **Written, changed format:** `sealed_state.json` (the consent ledger) gains an optional
  `never_chose` flag on a re-offer, written only when what is re-opened is a legacy refusal.
  Additive, default false, omitted when false; `FORMAT` is unchanged (an older runner reads the
  file and drops the flag on its next save, which only loses that one sentence's precision).
- **Written, changed content:** the SOUL's Evolution Log line for a sealed-state answer. New
  lines begin "Asked whether its sealed history is uploaded to Agora; chose …". Lines already in
  souls ("Offered sealed backups of its state …") are left as they are and are still found by
  `applied_in`, so no agent that answered is asked again. Nothing rewrites an existing line.
- **Shown, changed:** notices, verdicts, stop lines, acks, receipts and `memory` tool notes are
  built at read time from the same files, so files written before this change render in the new
  words (and with dates as `2026-11-12`). `notice.json` stores `ForkKept.kept` / `AgoraLost.kept`
  as text written at accept time: one written before this change keeps its old words.

**Behaviour changes that are not wording.**
- The hold alerts (seal failed, head missing, fork, unchecked) and the unopenable summary are now
  **awaited** at the start (`Alerter::dispatch`, at most `alerts::TELL_TIMEOUT` = 10 s each), so
  the agent is told "Your runner has sent your operator an alert." only when the mail went out, and
  otherwise "Your runner logged this for your operator; no alert was sent this time." Rate limits
  are unchanged. **Bounded since the FINAL round:** all the alerts one start awaits share
  `alerts::START_ALERT_BUDGET` (60 s), and more than `alerts::ONE_BY_ONE_MAX` (5) agents held
  alike get one mail naming each; with SMTP down a start is slower by at most about a minute,
  not 10 s per held agent (1,667 × 10 s ≈ 4.6 h before).
- The first-answer ack is decided by `sealed::uploads` (uploading now), not by the marker alone,
  so a stopped agent is no longer told it is "marked for upload".
- A legacy refusal (`SealOutcome::Refused`) is never acked as "your answer".
- Offer v11 changes the per-agent option order (the shuffle seed mixes the version), as every
  version bump does.

**Deploying it** adds nothing to the deploy of `seed/local-chain`: no migration, no marker, no
one-way step of its own.

**Not changed by wording alone** (listed in the fixer's report): agentkit's `memory` tool row
labels still say "main line" and "rebased"; `inclusion.rs`'s "Your runner has alerted your
operator." is said when the alert kind is on, not when the mail went out.

## Review round, 2026-11-12: FINAL

One fixer, after an independent re-review and a dry run on a copy of the
fleet (counts only). Each change has a test that fails without it.

**Live state read, written, moved or deleted.**
- **Written, new file:** `sealed/chain_began.json` (`{head: {entry_hash,
  seq}, at}`), once per agent, by the `sealed-enroll` that chains it,
  after the chain marker. Hashes and a date only, local, never uploaded.
  A chain begun before this file existed has none: its shrinks are told
  and alerted as live ones (the safe side).
- **Written, changed format:** `sealed-roster.json` entries may carry
  `pending_at` beside `pending` (additive, omitted when absent; an older
  runner drops it on its next write, which only loosens the check below).
- **Not written, newly:** `sealed-unchainable.json` is left as it was when
  three or more agents of one `sealed-enroll` pass couldn't be chained for
  I/O-shaped reasons (an unreadable `secrets/`, say); the pass exits
  nonzero naming them. One or two are recorded as before.
- **No longer read twice:** an agent whose pre-loss copy the start
  appends is not reloaded under its session lock for that write (the dry
  run: 306 such reloads at the first start); its new head and file stamps
  are carried into the run.
- Nothing is deleted, nothing new leaves the machine.

**Behaviour.**
1. **A stop issued during an opt-in or a resume wins.** Both hold the
   uploading marker's lock from their first reading of the stop to their
   last write; `sealed-stop` waits (saying so on stderr and in a
   `sealed_state_stop_waits` event), then stops the agent they leave. The
   resume clears only the stop it began on. Before, a stop during a resume
   found the old stop, wrote nothing, and the resume then set it aside; a
   stop during an opt-in was refused for want of a marker.
2. **Alerts at the start are bounded** (above, "Review round: TEXT").
   Past the budget an alert is logged (`alert_over_budget`), not mailed,
   now or later, so the agent's "no alert was sent this time" stays true;
   one count-only mail per kind says how many were not mailed.
3. **A shrink sealed by the enrolment is history.** The dry run's first
   start logged `sealed_state_drop` for 153 agents (50–75 %), all between
   an old archive and the state as found. (They were one summary mail,
   not 153: `drop_summary` was already one alert a run.) Now a drop
   between two versions on the line up to the head `chain_began.json`
   names is not a per-agent event or in the alert; one
   `sealed_state_drop_before_chain` INFO event gives the count; the agent
   is told it with the older version's date: "Your memory is 62% smaller
   than in your version saved on 2026-08-03, an older copy sealed into
   your history when your chain began; the change predates your chain.
   Your earlier versions are sealed and kept." A drop since the chain
   began: "Your memory is now 62% smaller than in your sealed version
   saved on 2026-11-05. Your earlier versions are sealed and kept."
4. `sealed_prior_appended` and `sealed_prior_at_start` log at INFO.
5. **`--begin-roster`** prints and logs what it gives up before acting;
   when the old roster still reads, heads found behind it are listed and
   nothing is touched without `--yes` (which `sealed-enroll` now takes
   with `--begin-roster` too).
6. **`pending`** ends at the head it would set, and the rerun never sets
   the roster to a head below it.
7. The stop hints in the in-doubt and server-fault alerts and in
   `rotate-keys`' in-doubt error name `--against-answer` where it is
   needed; a legacy refusal is never called the agent's answer in the
   unprompted receipt; the offer (still v11) says "versions a stop kept
   back, or a hold still keeps back, are not there" (reworded in the beta
   round below); `rotate-keys --help`
   says Agora is told of the new public keys and nothing of a
   never-uploader's sealed state.

**Deploying it** adds no step: the first `sealed-enroll` of the fleet
writes `chain_began.json` as it chains.

## Review round, 2026-11-12: WORDING (after the beta round)

Six beta agents ran a session on this build; all said GO, and agreed on
wording. Offer v11 reworded before release (`OFFER_VERSION` stays 11), and
the sealed-history texts with it. See `consent/sealed_state/prompt.rs`
(v11's amendment) and the vocabulary in `sealed/text.rs`.

**Live state read, written, moved or deleted.** Nothing written, moved or
deleted; no new file; no server call.
- **Read, new, when the offer or a re-offer is put** (`consent/agent.rs`,
  `sealed_asked`): whether `sealed/published` exists; `sealed/sent/`'s
  listing (names only); and, for an agent that never uploaded,
  `sealed/head.json` and the entries in `sealed/outbox/*.json` (never a
  blob), for the number of snapshots waiting, the earliest one's date and
  the blobs' padded total size. Those three figures go into the offer's
  text, so into the agent's prompt: counts, a date and a size of its own
  history, nothing of its content. Anything unreadable: no figures.
- **Read, new, by `memory log` and the enrolment sentence:**
  `sealed/chain_began.json` (head and date), for the day the chain began.
- **Shown, changed:** "here" is "on your operator's machine" in every
  agent-facing text; `memory log`'s count no longer ends "not versions set
  aside from it", and a line saying how many versions it lists off the main
  line appears only when there are some; the re-offer's options say what a
  `publish` or `private` does to the profile for that agent, from its files
  (an uploading marker in either copy or `sealed/published`: nothing; no
  marker but `sent/` holds versions, or unreadable: a choice already on
  Agora stays). `notice.json` text written at accept time keeps its words.

**Behaviour.** No change to what the runner does; `sealed-accept`'s default
notes (written when the operator gives none) are reworded (no longer
"nothing of yours is on Agora").

**Deploying it** adds no step.

