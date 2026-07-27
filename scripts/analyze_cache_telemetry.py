#!/usr/bin/env python3
"""Analyze `cache_proxy.py` telemetry into per-conversation chains.

The raw `requests.jsonl` is a flat sequence, but the meaningful unit is
the **conversation chain**: one agent's cycle, from its opening
single-message request through every follow-up turn. Deficits only mean
something *within* a chain.

Across a chain boundary the deficit is always large and positive — a new
agent starts fresh, so `cache_read` falls back to the `system + tools`
floor. That is a legitimate reset, not a cache failure, and treating it
as one is the single easiest way to misread this data (it has caused at
least three misdiagnoses across prior sessions). Chains are split on
`n_messages == 1`, and cross-boundary transitions are excluded.

What to look for, in increasing order of severity:

  * **Negative deficits** — healthy. The cache also holds the previous
    turn's output, so `cache_read` exceeds the previous `input_tokens`.
  * **Small positive deficits that recover** — a transient partial miss.
    Worth noting, not worth chasing on its own.
  * **Positive deficits that compound** (3 → 59 → 1115 → …) — the
    drama_llama#85 class. Once a prefix diverges the cache cannot
    resync, so the loss grows monotonically until it falls back to the
    floor. This is the pattern that matters.
  * **A pinned `cache_read`** — identical read across consecutive
    requests while `input_tokens` grows. The tail is not being retained.

Usage:
    ./analyze_cache_telemetry.py logs/run_*/telemetry_*/requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys


def load_chains(path: str) -> list[list[dict]]:
    """Split a flat request log into per-conversation chains."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    chains: list[list[dict]] = []
    current: list[dict] = []
    for r in rows:
        if r.get("n_messages") == 1:
            if current:
                chains.append(current)
            current = [r]
        else:
            current.append(r)
    if current:
        chains.append(current)
    return chains


def analyze(path: str) -> dict:
    chains = load_chains(path)
    rows = [r for c in chains for r in c]

    transitions = 0
    positives: list[tuple[int, int]] = []
    compounding: list[int] = []
    pinned: list[int] = []

    per_chain = []
    for i, chain in enumerate(chains):
        deficits = []
        for a, b in zip(chain, chain[1:]):
            prev_in = a["usage"].get("input_tokens") or 0
            read = b["usage"].get("cache_read_input_tokens") or 0
            deficits.append(prev_in - read)
        transitions += len(deficits)
        for d in deficits:
            if d > 0:
                positives.append((i, d))

        # Compounding: three or more consecutive positives, strictly
        # increasing. That is the signature that distinguishes a real
        # prefix collapse from a transient miss.
        run: list[int] = []
        for d in deficits:
            if d > 0 and (not run or d > run[-1]):
                run.append(d)
            else:
                if len(run) >= 3:
                    compounding.append(i)
                run = [d] if d > 0 else []
        if len(run) >= 3:
            compounding.append(i)

        # Pinned read: identical cache_read on consecutive requests while
        # input_tokens grows.
        for a, b in zip(chain, chain[1:]):
            ra = a["usage"].get("cache_read_input_tokens")
            rb = b["usage"].get("cache_read_input_tokens")
            ia = a["usage"].get("input_tokens") or 0
            ib = b["usage"].get("input_tokens") or 0
            if ra is not None and ra == rb and ib > ia:
                pinned.append(i)
                break

        per_chain.append({"index": i, "n": len(chain), "deficits": deficits})

    tot_in = sum(r["usage"].get("input_tokens") or 0 for r in rows)
    tot_read = sum(r["usage"].get("cache_read_input_tokens") or 0 for r in rows)

    return {
        "path": path,
        "requests": len(rows),
        "chains": len(chains),
        "transitions": transitions,
        "positives": positives,
        "compounding": sorted(set(compounding)),
        "pinned": sorted(set(pinned)),
        "total_input": tot_in,
        "total_read": tot_read,
        "hit_rate": (100.0 * tot_read / tot_in) if tot_in else 0.0,
        "per_chain": per_chain,
        "model": next((r.get("model") for r in rows if r.get("model")), None),
    }


def report(a: dict, verbose: bool) -> None:
    print(f"== {a['path']}")
    print(f"   model      {a['model']}")
    print(f"   requests   {a['requests']}  chains {a['chains']}")
    print(
        f"   cache hit  {a['total_read']}/{a['total_input']} = {a['hit_rate']:.1f}%"
    )
    print(
        f"   deficits   {len(a['positives'])} positive "
        f"of {a['transitions']} within-chain transitions"
        + (f", max +{max(d for _, d in a['positives'])}" if a["positives"] else "")
    )

    if a["compounding"]:
        print(
            f"   COMPOUNDING in chain(s) {a['compounding']} "
            "— drama_llama#85 class, prefix cannot resync"
        )
    if a["pinned"]:
        print(
            f"   PINNED cache_read in chain(s) {a['pinned']} "
            "— tail not retained as the conversation grows"
        )
    if not a["compounding"] and not a["pinned"]:
        print("   no compounding runs, no pinned reads")

    if verbose:
        print()
        print("   chain   n  within-chain deficits")
        for c in a["per_chain"]:
            print(f"   {c['index']:5d} {c['n']:3d}  {c['deficits']}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="requests.jsonl file(s)")
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="print per-chain deficit lists"
    )
    args = ap.parse_args()

    analyses = []
    for p in args.paths:
        try:
            a = analyze(p)
        except FileNotFoundError:
            print(f"!! {p}: not found", file=sys.stderr)
            continue
        analyses.append(a)
        report(a, args.verbose)

    if len(analyses) > 1:
        print("== comparison")
        width = max(len(str(a["model"])) for a in analyses)
        for a in analyses:
            flags = []
            if a["compounding"]:
                flags.append("COMPOUNDING")
            if a["pinned"]:
                flags.append("PINNED")
            print(
                f"   {str(a['model']):<{width}}  {a['hit_rate']:5.1f}%  "
                f"{len(a['positives'])}/{a['transitions']} positive  "
                + (" ".join(flags) or "clean")
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
