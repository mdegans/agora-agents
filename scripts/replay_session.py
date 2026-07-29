#!/usr/bin/env python3
"""Replay a captured blallama session and report where the cache misses.

Two modes, both reading what `cache_proxy.py` wrote:

  # the whole session, front to back, in capture order
  ./replay_session.py --turns telemetry_tipfix/turns --endpoint http://127.0.0.1:11435

  # just one COLLAPSE pair: primer, then the request that missed
  ./replay_session.py --pair telemetry_tipfix/COLLAPSE_0015_*.json \
      --endpoint http://127.0.0.1:11435

Why a pair and not a single request: a stall is a property of *two*
requests — the one that primed the cache and the one that failed to reuse
it. Replayed alone, the missing request is just an ordinary cold start and
reproduces nothing at all.

The check is `deficit = prev.input_tokens - this.cache_read`. Negative is
healthy (the cache also holds the prior turn's output). Positive means the
server failed to reuse something it already had. Requests whose message
list has length 1 are fresh conversations, whose deficit is meaningless —
they are reported as `boundary` and excluded from the miss count.

Exit status is 1 if any real miss reproduced, so this can gate a bisect.

Stdlib only, by design — this runs wherever the model runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request


def post(endpoint: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/messages",
        data=data,
        headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": {"message": e.read().decode("utf-8", "replace")[:400]}}


def usage_of(resp: dict) -> tuple[int, int, int]:
    u = resp.get("usage") or {}
    return (
        u.get("input_tokens") or 0,
        u.get("cache_read_input_tokens") or 0,
        u.get("cache_creation_input_tokens") or 0,
    )


def replay(requests: list[tuple[str, dict, dict | None]], endpoint: str, timeout: float) -> int:
    """requests: list of (label, request_body, original_record or None)."""
    prev_input = None
    misses = 0
    print(
        f"{'#':>4} {'label':22s} {'nmsg':>5} {'in':>7} {'read':>7} "
        f"{'create':>7} {'deficit':>8}  {'orig':>8}"
    )
    for i, (label, body, orig) in enumerate(requests):
        t0 = time.time()
        resp = post(endpoint, body, timeout)
        if "error" in resp and "usage" not in resp:
            print(f"{i:4d} {label[:22]:22s} ERROR {resp['error'].get('message','')[:80]}")
            prev_input = None
            continue
        inp, read, create = usage_of(resp)
        nmsg = len(body.get("messages") or [])
        boundary = nmsg == 1
        deficit = None if (prev_input is None or boundary) else prev_input - read
        orig_def = (orig or {}).get("deficit_same_model")
        d = "boundary" if boundary else ("n/a" if deficit is None else f"{deficit:+d}")
        o = "-" if orig_def is None else f"{orig_def:+d}"
        flag = ""
        if deficit is not None and deficit > 0:
            misses += 1
            flag = "  <-- MISS"
        print(
            f"{i:4d} {label[:22]:22s} {nmsg:5d} {inp:7d} {read:7d} "
            f"{create:7d} {d:>8}  {o:>8}{flag}   ({time.time()-t0:.1f}s)"
        )
        prev_input = inp
    return misses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--turns", help="turns/ directory written by cache_proxy.py")
    src.add_argument("--pair", help="a COLLAPSE_*.json to replay as primer + miss")
    ap.add_argument("--endpoint", default="http://127.0.0.1:11435")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--model", help="only replay turns for this model")
    ap.add_argument("--limit", type=int, help="stop after N requests")
    args = ap.parse_args()

    items: list[tuple[str, dict, dict | None]] = []
    if args.pair:
        d = json.load(open(args.pair))
        if not d.get("prev_request"):
            print(
                f"{args.pair} has no prev_request — it was the first request of "
                "its run, so there is no primer and nothing to reproduce.",
                file=sys.stderr,
            )
            return 2
        items.append(("primer", d["prev_request"], d.get("prev_record")))
        items.append(("MISS", d["request"], d.get("record")))
    else:
        for p in sorted(glob.glob(os.path.join(args.turns, "*.json"))):
            d = json.load(open(p))
            req = d.get("request")
            if not req:
                continue
            if args.model and req.get("model") != args.model:
                continue
            items.append((os.path.basename(p)[:22], req, d.get("record")))
            if args.limit and len(items) >= args.limit:
                break

    if not items:
        print("nothing to replay", file=sys.stderr)
        return 2

    print(f"replaying {len(items)} request(s) against {args.endpoint}\n")
    misses = replay(items, args.endpoint, args.timeout)
    print(f"\n{misses} miss(es) reproduced")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
