#!/usr/bin/env python3
"""Replay a COLLAPSE_*.json dump against a blallama endpoint.

A prefix-cache stall is a property of a *pair* of requests: one that
primes the cache and one that fails to reuse it. Replaying the
collapsing request alone just produces a cold start, which reproduces
nothing. So this script sends the primer first, then the collapsing
request, and reports what the cache actually did.

Dumps written before 2026-07-28 have no `prev_request` field (the proxy
only kept the previous request's *token count*, not its body). For
those, pass --synth-primer to reconstruct an approximate primer by
dropping the trailing message from the collapsing request. That shares
the prefix that matters and is usually enough to show the stall, but it
is *not* byte-identical to what the server originally saw — say so when
attaching results to a bug report.

Healthy:  the second request reads back nearly all of the first's
          input_tokens, i.e. `advance` is large and `deficit` negative.
Stalled:  `advance` is ~0 while the prompt grew by hundreds of tokens.

Stdlib only — this is meant to run anywhere, including on the machine
hosting the model, with no install step.

Usage:
    ./replay_collapse.py COLLAPSE_0010_*.json --endpoint http://192.168.0.123:11435
    ./replay_collapse.py COLLAPSE_0010_*.json --synth-primer
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request


def post(endpoint: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/messages",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            resp = json.loads(fh.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", "replace")[:2000]
        print(f"  HTTP {e.code}: {body_text}", file=sys.stderr)
        raise
    return resp, time.monotonic() - t0


def usage_of(resp: dict) -> tuple[int, int, int]:
    u = resp.get("usage") or {}
    return (
        u.get("input_tokens") or 0,
        u.get("cache_read_input_tokens") or 0,
        u.get("cache_creation_input_tokens") or 0,
    )


def synth_primer(request: dict) -> dict | None:
    """Approximate primer: the same request minus its trailing message."""
    msgs = request.get("messages") or []
    if len(msgs) < 2:
        return None
    primer = copy.deepcopy(request)
    primer["messages"] = msgs[:-1]
    # The truncated tail must end on a user turn to be a valid request.
    while primer["messages"] and primer["messages"][-1].get("role") != "user":
        primer["messages"].pop()
    return primer if primer["messages"] else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="COLLAPSE_*.json written by cache_proxy.py")
    ap.add_argument(
        "--endpoint",
        default="http://192.168.0.123:11435",
        help="blallama base URL (default: %(default)s)",
    )
    ap.add_argument(
        "--synth-primer",
        action="store_true",
        help="reconstruct the primer by dropping the trailing message "
        "(for dumps with no prev_request)",
    )
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    with open(args.dump, encoding="utf-8") as fh:
        dump = json.load(fh)

    request = dump["request"]
    primer = dump.get("prev_request")
    origin = "captured"
    if primer is None:
        if not args.synth_primer:
            print(
                "This dump has no `prev_request` (written by an older "
                "cache_proxy.py). Re-run with --synth-primer to "
                "approximate it, and note the approximation in any bug "
                "report.",
                file=sys.stderr,
            )
            return 2
        primer = synth_primer(request)
        origin = "synthesized (NOT byte-identical to the original)"
        if primer is None:
            print("Cannot synthesize a primer from this request.", file=sys.stderr)
            return 2

    rec = dump.get("record") or {}
    print(f"dump      : {args.dump}")
    print(f"endpoint  : {args.endpoint}")
    print(f"model     : {request.get('model')}")
    print(f"primer    : {origin}, {len(primer.get('messages') or [])} messages")
    print(f"collapsing: {len(request.get('messages') or [])} messages")
    print(f"originally: in={rec.get('usage', {}).get('input_tokens')} "
          f"read={rec.get('usage', {}).get('cache_read_input_tokens')} "
          f"deficit={rec.get('deficit')}")
    print()

    print("[1/2] priming ...")
    p_resp, p_secs = post(args.endpoint, primer, args.timeout)
    p_in, p_read, p_creat = usage_of(p_resp)
    print(f"      in={p_in} read={p_read} create={p_creat} ({p_secs:.0f}s)")

    print("[2/2] replaying the collapsing request ...")
    c_resp, c_secs = post(args.endpoint, request, args.timeout)
    c_in, c_read, c_creat = usage_of(c_resp)
    print(f"      in={c_in} read={c_read} create={c_creat} ({c_secs:.0f}s)")

    advance = c_read - p_read
    deficit = p_in - c_read
    growth = c_in - p_in
    print()
    print(f"prompt growth : {growth:+d}")
    print(f"cache advance : {advance:+d}   (read moved this much)")
    print(f"deficit       : {deficit:+d}   (prev.input_tokens - this.cache_read)")
    print()
    # The advance is the sharper signal: a stall can hide under a deficit
    # threshold when the prompt happened not to grow much.
    if advance < 10 and growth > 100:
        print("VERDICT: STALL REPRODUCED — the prompt grew but the cache "
              "did not advance.")
        return 1
    if deficit > 0:
        print("VERDICT: positive deficit, but the cache did advance. "
              "Weaker signal; compare against the original numbers above.")
        return 1
    print("VERDICT: healthy — no stall on this replay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
