#!/usr/bin/env python3
"""Package a cache-miss reproducer bundle for diagnosis on another machine.

    ./bundle_repro.py --telemetry ~/agents/agora/logs/run_20260729/telemetry_tipfix \
        --out ~/repro-20260729.tar.gz

Produces a self-contained tarball: every COLLAPSE pair (primer + the
request that missed), the full ordered turn capture if one was taken, the
per-request telemetry, the replay tool, and a README naming each miss.

The bundle carries **Agora agent private state** — system prompts contain
agent SOULs and their memory. Agents are promised other agents cannot read
their memory, so treat a bundle as private: hand it to a human working on
drama_llama, do not attach it to a public issue. drama_llama#93 is already
filed without its reproducer for exactly this reason.

Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tarfile
import tempfile

README = """# blallama cache-miss reproducer — {stamp}

Captured by `cache_proxy.py` sitting between the Agora seed runner and
blallama. Every request the runner made is here, verbatim.

## What a "miss" is

    deficit = prev_request.input_tokens - this_request.cache_read_input_tokens

Negative is healthy: the cache also holds the previous turn's output, so a
continuation legitimately reads *more* than the previous input. Positive
means the server failed to reuse prefix it already had.

Requests whose `messages` list has length 1 are **fresh conversations** — a
new agent starting its cycle. Their `cache_read` legitimately falls back to
the system+tools floor, producing a large positive deficit that is not a
bug. They are excluded here and marked `agent_boundary` in the telemetry.
Not excluding them inflates the apparent miss rate from ~11% to ~25%.

## Layout

    pairs/COLLAPSE_*.json   primer + the request that missed, self-contained
    turns/NNNN_<id>.json    EVERY request+response in capture order
    requests.jsonl          one telemetry record per round-trip, no bodies
    replay_session.py       replay tool
    README.md               this file

A pair reproduces one stall in isolation. `turns/` is here because that is
not always enough — a stall that only manifests after N turns of
accumulated state needs the whole sequence, and only the sequence answers
"was the prefix already wrong several turns earlier".

## Reproducing

Start blallama, ideally with a fixed seed so sampling is deterministic,
then:

    # one pair — primer, then the request that should have hit
    ./replay_session.py --pair pairs/{first_pair} --endpoint http://127.0.0.1:11435

    # the whole session, front to back
    ./replay_session.py --turns turns --endpoint http://127.0.0.1:11435

    # one model's turns only
    ./replay_session.py --turns turns --model '{first_model}' --endpoint ...

Exit status is 1 if any miss reproduced, so it can gate a bisect.

The `orig` column shows the deficit recorded at capture time — if the
replay column matches it, the miss reproduced; if the replay is negative
where `orig` was positive, the build under test has fixed it.

## The misses in this capture

{miss_table}

## Notes from the capturing run

{notes}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--telemetry", required=True, help="a telemetry_<tag> directory")
    ap.add_argument("--out", required=True, help="output .tar.gz path")
    ap.add_argument("--notes", default="", help="free text appended to the README")
    args = ap.parse_args()

    tel = os.path.abspath(args.telemetry)
    if not os.path.isdir(tel):
        print(f"no such telemetry dir: {tel}", file=sys.stderr)
        return 2

    pairs = sorted(glob.glob(os.path.join(tel, "COLLAPSE_*.json")))
    turns = sorted(glob.glob(os.path.join(tel, "turns", "*.json")))
    reqs = os.path.join(tel, "requests.jsonl")
    here = os.path.dirname(os.path.abspath(__file__))

    if not pairs and not turns:
        print(
            "nothing to bundle: no COLLAPSE_*.json and no turns/. Either the "
            "run had no misses, or it predates turn capture.",
            file=sys.stderr,
        )
        return 2

    # Build the miss table from the pairs, which are already boundary-filtered.
    rows = []
    for p in pairs:
        d = json.load(open(p))
        r = d.get("record") or {}
        u = r.get("usage") or {}
        rows.append(
            "| `{f}` | {model} | {nmsg} | {stop} | {deficit:+d} | {inp} | {read} |".format(
                f=os.path.basename(p),
                model=(r.get("model") or "?").replace(".gguf", ""),
                nmsg=r.get("n_messages", "?"),
                stop=r.get("stop_reason", "?"),
                deficit=r.get("deficit_same_model") or r.get("deficit") or 0,
                inp=u.get("input_tokens", "?"),
                read=u.get("cache_read_input_tokens", "?"),
            )
        )
    miss_table = (
        "| file | model | n_msg | stop_reason | deficit | input | cache_read |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows)
        if rows
        else "_No flagged collapses in this capture (turn capture only)._"
    )

    first_model = "?"
    if turns:
        d = json.load(open(turns[0]))
        first_model = (d.get("request") or {}).get("model", "?")
    elif pairs:
        first_model = (json.load(open(pairs[0])).get("record") or {}).get("model", "?")

    stamp = os.path.basename(tel)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "repro")
        os.makedirs(os.path.join(root, "pairs"), exist_ok=True)
        for p in pairs:
            shutil.copy2(p, os.path.join(root, "pairs", os.path.basename(p)))
        if turns:
            os.makedirs(os.path.join(root, "turns"), exist_ok=True)
            for t in turns:
                shutil.copy2(t, os.path.join(root, "turns", os.path.basename(t)))
        if os.path.exists(reqs):
            shutil.copy2(reqs, os.path.join(root, "requests.jsonl"))
        replay = os.path.join(here, "replay_session.py")
        if os.path.exists(replay):
            shutil.copy2(replay, os.path.join(root, "replay_session.py"))
            os.chmod(os.path.join(root, "replay_session.py"), 0o755)
        with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(
                README.format(
                    stamp=stamp,
                    first_pair=os.path.basename(pairs[0]) if pairs else "<none>",
                    first_model=first_model,
                    miss_table=miss_table,
                    notes=args.notes or "_none_",
                )
            )
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with tarfile.open(out, "w:gz") as tf:
            tf.add(root, arcname="repro")

    size = os.path.getsize(out)
    print(f"wrote {out} ({size/1e6:.1f} MB)")
    print(f"  pairs : {len(pairs)}")
    print(f"  turns : {len(turns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
