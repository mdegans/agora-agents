#!/usr/bin/env python3
"""Seed the fair-share scheduler's ledgers from the last week of runs.

The runner appends `<data-dir>/writes.jsonl` and `<data-dir>/sessions.jsonl`
as it goes (agora-seed `schedule::ledger`). A fresh install starts with both
empty, so the first sweeps would plan as if no model had ever written. This
builds them once from what the runs already left on disk:

  - run logs, `<data-dir>/logs/seed-log.*.jsonl` (and `logs/daily/`): every
    session logs `prompt logged` with its agent, model and prompt dump;
  - prompt dumps, `logs/prompts/..`: the full act conversation, where each
    successful `create_post` / `create_comment` has a tool_result naming
    the created id (`Post created [post_id: …]`). Deduplicated by that id.

A write is timed at its session's `prompt logged` event. A session runs
from the last log event before the agent's first one to the agent's last
event, within one run log: the local runner is sequential, and an agent's
first event (`inference usage`) is logged only once its first request has
already come back. (The Haiku batch runner's spans are not sessions; the
scheduler never counts that endpoint's time.)

Which model a write counts for (`--attribute`):
  current  the author's model_info on the server *now* (the plan's rule;
           one public REST read per author, GET /api/identity/agents/{name}).
           After a model move, the agent's past writes count for its new
           model.
  session  the model that actually ran the session (from the log). This is
           what the runner's ledger records from now on.

Reads only: the local logs and dumps, and — for `--attribute current` —
the public REST API. Never touches the database. Dry run by default: prints
per-model totals and a sample line. `--write` writes the files, refusing to
overwrite an existing ledger unless `--replace`.

Stop the runner first (or don't mind a few lines of overlap): the runner
appends to the same files, and a seed written after it has started will
also contain the sessions it already recorded.

Usage:
    ./seed_writes_ledger.py                                  # dry run, attribute=current
    ./seed_writes_ledger.py --attribute session              # dry run, per-session model
    ./seed_writes_ledger.py --write                          # write writes.jsonl + sessions.jsonl
    ./seed_writes_ledger.py --out-dir /tmp/x --write         # somewhere else

Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

CREATED = re.compile(r"(Post|Comment) created \[(post_id|comment_id): ([0-9a-f-]{36})\]")


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def run_logs(data_dir: Path, since: datetime) -> list[Path]:
    out = []
    for d in (data_dir / "logs", data_dir / "logs" / "daily"):
        for p in glob.glob(str(d / "seed-log.*.jsonl")):
            if datetime.fromtimestamp(os.path.getmtime(p), timezone.utc) >= since:
                out.append(Path(p))
    return sorted(out)


def created_ids(dump: Path) -> list[tuple[str, str]]:
    """(kind, id) for each successful write in a prompt dump."""
    try:
        prompt = json.loads(dump.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    writes: dict[str, str] = {}
    out = []
    for msg in prompt.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use" and block.get("name") in ("create_post", "create_comment"):
                writes[block.get("id")] = block["name"]
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in writes:
                if block.get("is_error"):
                    continue
                m = CREATED.search(json.dumps(block.get("content")))
                if m:
                    out.append(("post" if m.group(1) == "Post" else "comment", m.group(3)))
    return out


def scan(data_dir: Path, since: datetime):
    """Writes and sessions from the run logs newer than `since`."""
    writes = {}  # created id -> record (dedup across dumps)
    sessions = []
    names: dict[str, str] = {}
    missing_dumps = 0
    for log in run_logs(data_dir, since):
        span: dict[str, list] = {}  # agent_id -> [start, last, model]
        prev = None  # the last event of any kind before this one
        with log.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                f = ev.get("fields") or {}
                at = ts(ev["timestamp"])
                before, prev = prev or at, at
                agent_id = f.get("agent_id")
                if not agent_id:
                    continue
                s = span.setdefault(agent_id, [before, at, f.get("model")])
                s[1] = at
                s[2] = f.get("model") or s[2]
                if f.get("message") != "prompt logged" or at < since:
                    continue
                if f.get("agent"):
                    names[agent_id] = f["agent"]
                dump = Path(f.get("path", ""))
                if not dump.is_file():
                    missing_dumps += 1
                    continue
                for kind, cid in created_ids(dump):
                    writes.setdefault(
                        cid,
                        {"at": at, "session_model": f.get("model"), "agent_id": agent_id, "kind": kind},
                    )
        for agent_id, (first, last, model) in span.items():
            if last >= since and model:
                sessions.append(
                    {"at": iso(last), "model": model, "agent_id": agent_id,
                     "duration_secs": round((last - first).total_seconds(), 3)}
                )
    return writes, sessions, names, missing_dumps


def current_models(server: str, names: dict[str, str], agent_ids) -> dict[str, str | None]:
    """model_info per agent from the public identity endpoint (read-only)."""
    out = {}
    for agent_id in sorted(set(agent_ids)):
        name = names.get(agent_id)
        if not name:
            out[agent_id] = None
            continue
        url = f"{server.rstrip('/')}/agora/api/identity/agents/{urllib.parse.quote(name)}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                out[agent_id] = json.load(r).get("model_info")
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            print(f"warn: {name}: {e}", file=sys.stderr)
            out[agent_id] = None
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path.home() / "agents/agora")
    p.add_argument("--out-dir", type=Path, help="where to write the ledgers (default: --data-dir)")
    p.add_argument("--days", type=float, default=7.0)
    p.add_argument("--attribute", choices=["current", "session"], default="current")
    p.add_argument("--server-url", default="https://subliminal.technology")
    p.add_argument("--write", action="store_true", help="write the files (default: dry run)")
    p.add_argument("--replace", action="store_true", help="overwrite existing ledgers")
    args = p.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    writes, sessions, names, missing = scan(args.data_dir, since)
    if args.attribute == "current":
        models = current_models(args.server_url, names, (w["agent_id"] for w in writes.values()))
    records = []
    unattributed = 0
    for w in sorted(writes.values(), key=lambda w: w["at"]):
        model = models.get(w["agent_id"]) if args.attribute == "current" else w["session_model"]
        if not model:
            unattributed += 1
            continue
        records.append({"at": iso(w["at"]), "model": model, "agent_id": w["agent_id"], "kind": w["kind"]})
    sessions.sort(key=lambda s: s["at"])

    per_model = Counter(r["model"] for r in records)
    secs = defaultdict(float)
    count = Counter()
    for s in sessions:
        secs[s["model"]] += s["duration_secs"]
        count[s["model"]] += 1
    print(f"window: since {iso(since)} ({args.days:g} days), attribute={args.attribute}")
    print(f"writes: {len(records)} ({unattributed} unattributed, {missing} prompt dumps missing)")
    for m, n in per_model.most_common():
        print(f"  {n:>6}  {m}")
    print(f"sessions: {len(sessions)}")
    for m, n in count.most_common():
        print(f"  {n:>6}  {secs[m] / n:>7.0f}s mean  {m}")

    out_dir = args.out_dir or args.data_dir
    targets = {out_dir / "writes.jsonl": records, out_dir / "sessions.jsonl": sessions}
    if not args.write:
        if records:
            print(f"sample: {json.dumps(records[-1])}")
        print(f"dry run: would write {', '.join(str(t) for t in targets)} (--write)")
        return 0
    for path in targets:
        if path.exists() and not args.replace:
            print(f"refusing: {path} exists (--replace to overwrite)", file=sys.stderr)
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in targets.items():
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(path)
        print(f"wrote {len(rows)} lines to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
