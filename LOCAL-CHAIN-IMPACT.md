# Local chain for every agent: live-state impact statement

Plan: agora `memory/project_local_chain_for_all.md` (2026-11-11). Every agent
is chained locally; consent decides only whether the chain is uploaded.
Built in steps B1-B7 on the integration branch `seed/local-chain`; each step
appends its section. Paths: `<state>` is `<data_dir>/state`, `<secrets>` is
`<data_dir>/secrets`, `<id>` an agent id.

## B1: markers, API guard, storage, drain, stop, migration

Nothing in B1 deletes or moves a file. Installing the binary changes no
agent file by itself (no file in any repository tree holds agent state).
Everything below happens when the runner, or a `sealed-*` / `rotate-keys`
command, first runs after install.

### Files read

- `<state>/` and `<secrets>/` listings (directory names).
- `<state>/<id>/sealed/enrolled`: existence only.
- `<state>/<id>/sealed/uploading` and `<secrets>/<id>/sealed_uploading`
  (new): parsed (`{agent_id, since, by}`), on every upload decision.
- `<state>/<id>/sealed/stopped` and `<secrets>/<id>/sealed_stopped`:
  existence, and parsed as before.
- `<state>/<id>/sealed/sent/`: whether it has any entry (the first entry's
  name only), for the migration.
- `<state>/<id>/sealed_state.json` (the consent ledger): parsed for its
  answer (yes / no), for the migration.
- `<state>/<id>/state.json`: only for a stop written before B1 (below), to
  hash and seal it, as any save does. Its content is never logged.
- Keys in `<secrets>/<id>/` (signing key, state key): as any seal does,
  for that same catch-up.

### Files written

| File | When | How |
|---|---|---|
| `<state>/<id>/sealed/uploading` | migration (b, c below); `sealed-enroll` of an agent that said yes, once its chain is sealed; put back from the secrets copy if lost | atomic: unique `uploading.<uuid>.writing`, fsync, rename, fsync of `sealed/`; 0600 |
| `<secrets>/<id>/sealed_uploading` | same, written second (it commits the fact) | same |
| `<state>/<id>/sealed/stopped`, `<secrets>/<id>/sealed_stopped` | `sealed-stop` (now with `"chained": true`); the catch-up of a pre-B1 stop rewrites both with `"chained": true` | now atomic with a unique temporary and 0600 (was a fixed `stopped.tmp`) |
| `<state>/<id>/sealed/{blobs,outbox,head.json,…}` | every save of a **stopped** agent now seals and chains (agentkit `Mirrored`), where it used to save plainly; the pre-B1-stop catch-up seals one version | agentkit, under `sealed/lock` |
| `<state>/<id>/state.json` | a stopped agent's saves are now written as format 2 (marked), not format 1 | agentkit `FsStorage` |
| `sealed/refused`, `notice.json`, sizes log | the start check now also checks stopped agents, and records and refuses as for any enrolled agent | as before |

No longer written:

- `<state>/<id>/sealed/receipts.json` for an agent that doesn't upload: the
  drainer's baseline and first-send records are made only for uploaders,
  and first sends only from agentkit's `BeforeUpload` hook (a drain that
  will send). A stopped agent used to get observed acknowledgements at each
  run start. (`sealed-enroll`, `sealed-stop` and `rotate-keys` still take a
  baseline for the agent they act on, which for a stopped agent can record
  observed acknowledgements of versions already on Agora: local only.)
- No anchor work (`anchors.json`, `inclusion.json`, `published`) and no
  network call of any kind for an agent that doesn't upload: agentkit's
  drain gate, and `sealed::api::UploadGuard` on every per-agent client
  (the run's, the sealed commands', the memory tool's).

Via agentkit#294, not B1's code but in this build: `FsStorage` archives
`state.json` by hard link (never absent mid-save), sets a leftover
`state.json.tmp` aside as `state.json.tmp.<utc-ms>.aside` instead of
truncating it, and `Mirrored` seals every enrolled agent whether or not it
uploads.

### The first run after install

**(a) An agent not enrolled** (the whole fleet today). One `stat` of
`sealed/enrolled` (the migration); nothing written. Saves go plainly, as
before (agentkit gives an agent with no `sealed/enrolled` no `sealed/`). Not
in the start check. No change.

**(b) An enrolled agent that uploads** (the three agora-sim betas). The
migration finds it enrolled, unmarked, with something in `sealed/sent/` and
its ledger saying yes: it writes `sealed/uploading`, then
`<secrets>/<id>/sealed_uploading`, and logs one
`sealed_state_uploading_migrated` event. From then on, as before: chained,
checked, drained, anchored. Idempotent: later runs see the marker and write
nothing. *Gap:* an enrolled agent whose ledger says yes but whose first
upload never succeeded (empty `sent/`) is **not** migrated, by the rule in
the brief, and silently stops uploading; `sealed-enroll` skips it as
enrolled. Until B3's opt-in step exists the only fix is the steward's. Check
`sealed/sent/` of the three betas (counts only) before installing.

**(c) An enrolled agent whose uploads were stopped.** The migration marks
it uploading if it had uploaded (a stop always drained first, so `sent/` is
non-empty) and its ledger says yes; it still doesn't upload while stopped.
Its stop, written before B1, has no `chained`: at the run's start the
catch-up takes its `sealed/lock` (skipped, and retried next start, while a
session journal exists), and if the state the plain saves left differs from
its head, seals it as one local version (`sealed_state_stop_caught_up`),
then marks the stop `chained` (secrets copy first). It is then **checked at
every start** like any enrolled agent (the stop's line first, then the
verdict, which in B1 still reads "Agora not checked": B2 rewords it). Its
saves chain locally; nothing of it is drained, asked or sent. A refusal
(a mismatch, a missing head, a catch-up that couldn't run while its head no
longer matches) now refuses the whole run (exit 75), where a stopped agent
used to be skipped. `sealed-enroll --agent <id>` (resume) is refused while
any version saved while stopped waits (B3/B5 decide what a resume uploads).

**(d) An agent that declined.** Not enrolled (`sealed-enroll` enrols only a
yes), so as (a). Should one be enrolled with a ledger that says no, the
migration never marks it: it is chained and checked, and nothing uploads;
before B1 it would have uploaded.
