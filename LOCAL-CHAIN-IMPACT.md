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

## B3: local chain for every agent, opt-in, resume, re-offer

Nothing in B3 deletes a file. Installing changes no agent file by itself;
everything below happens only when an operator runs a command (or, for
provisioning, creates an agent). Every `sealed-enroll` (but `--dry-run`)
and `sealed-reoffer` takes `<data_dir>/runner.lock` **exclusively, first,
for the whole command, without waiting**: refused while a runner runs
(`just runner-stop` first); only then does B1's start pass run.

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
  two is finished by `Enroller::finish_prior` (to be called at run start).
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
else wrapped afresh); writes the agent's notice (one write); marks it
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
