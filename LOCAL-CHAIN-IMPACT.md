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
  unreadable, I/O, too new) is held back; a memory that shrank by more than
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

Call sites marked `// B2/B5: roster::replace` (accept, rebase, the Agora-loss
resolution, `--keep`, restore, `--from-cut-save`, forget with `None`); not
called until the roster merges.
