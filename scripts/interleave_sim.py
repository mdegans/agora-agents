#!/usr/bin/env python3
"""Simulate `interleave` (agora-seed/src/main.rs) run order for a candidate roster.

The interleave chunks each model cohort into k = ceil(n / wave_size) waves and
sorts every wave from every cohort by (i + 0.5) / k. That spreads each cohort
proportionally across the run, but it does NOT bound how many same-model waves
land back to back — with a lopsided roster you get long single-model blocks,
which is exactly the homogeneous conversation cluster the interleave exists to
prevent (agents read and reply to the feed).

Use this before a sweep to price a roster or a wave_size against the block
length it will actually produce. Verified against the 2026-07-29 sweep: it
reproduces the observed distribution exactly (max cogito block 40 at
wave_size=8). See memory/project_interleave_clustering_2026_07_31.md.

Usage:
    ./interleave_sim.py                                  # current roster, wave_size sweep
    ./interleave_sim.py cogito=400 gemma=400 qwen=400    # candidate roster
    ./interleave_sim.py --wave-size 4 cogito=1117 ...
    ./interleave_sim.py --focus gpt-oss                  # block stats for another cohort
"""

import argparse
import math
import sys

# The 2026-07-31 blallama-routed fleet. Overridable on the command line.
DEFAULT_ROSTER = {
    "cogito": 1117,
    "gpt-oss": 246,
    "qwen": 164,
    "mistral": 143,
}

# Mean first-request penalty after a model switch, measured from 07-29
# telemetry (cogito 109s, gpt-oss 63s, Qwen 23s). Only ~10-15s of that is
# model load; the rest is re-prefilling the ~7700-token floor cold.
SWITCH_PENALTY_SECS = 64.8

# Measured sweep rate. Dense cogito is far slower than the MoE models
# (3.28 vs 1.00 min/agent), so this flat average is for rough sizing only —
# see memory/project_full_fleet_sweep_2026_07_29.md.
MINS_PER_AGENT = 2.5


def run_order(roster, wave_size):
    """Return the merged wave order as a list of (model, wave_size) pairs."""
    wave_size = max(1, wave_size)
    waves = []
    # BTreeMap iteration in the Rust source is by model name, so sort to match.
    for name, n in sorted(roster.items()):
        if n <= 0:
            continue
        k = max(1, math.ceil(n / wave_size))
        for i in range(k):
            # Near-even split: the first n % k waves get one extra.
            size = n // k + (1 if i < n % k else 0)
            waves.append(((i + 0.5) / k, name, size))
    waves.sort(key=lambda w: w[0])
    return [(name, size) for _, name, size in waves]


def blocks(order):
    """Collapse the run order into contiguous same-model blocks."""
    out = []
    for name, size in order:
        if out and out[-1][0] == name:
            out[-1][1] += size
        else:
            out.append([name, size])
    return out


def analyze(roster, wave_size, focus):
    blks = blocks(run_order(roster, wave_size))
    sizes = [b[1] for b in blks if b[0] == focus]
    total = sum(roster.values())
    switches = len(blks) - 1
    return {
        "total": total,
        "share": roster.get(focus, 0) / total * 100 if total else 0.0,
        "switches": switches,
        "switch_hours": switches * SWITCH_PENALTY_SECS / 3600,
        "sweep_hours": total * MINS_PER_AGENT / 60,
        "max_block": max(sizes) if sizes else 0,
        "mean_block": sum(sizes) / len(sizes) if sizes else 0.0,
        "block_hist": {s: sizes.count(s) for s in sorted(set(sizes))},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Simulate agora-seed interleave block structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    ap.add_argument(
        "roster",
        nargs="*",
        metavar="MODEL=N",
        help="cohort sizes; defaults to the 2026-07-31 fleet",
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        help="single wave_size to analyze; omit to sweep 8/6/4/3/2",
    )
    ap.add_argument(
        "--focus",
        default=None,
        help="cohort to report block stats for (default: the largest)",
    )
    args = ap.parse_args()

    roster = dict(DEFAULT_ROSTER)
    if args.roster:
        roster = {}
        for item in args.roster:
            if "=" not in item:
                sys.exit(f"expected MODEL=N, got {item!r}")
            name, _, count = item.partition("=")
            try:
                roster[name] = int(count)
            except ValueError:
                sys.exit(f"not an integer count: {item!r}")

    if not roster or sum(roster.values()) <= 0:
        sys.exit("roster is empty")

    focus = args.focus or max(roster, key=roster.get)
    if focus not in roster:
        sys.exit(f"--focus {focus!r} is not in the roster: {sorted(roster)}")

    sizes = ", ".join(f"{k}={v}" for k, v in sorted(roster.items()))
    print(f"roster: {sizes}  (N={sum(roster.values())})")
    print(f"focus:  {focus}  ({roster[focus] / sum(roster.values()) * 100:.0f}% of fleet)\n")

    wave_sizes = [args.wave_size] if args.wave_size else [8, 6, 4, 3, 2]
    hdr = f"{'wave_size':>10}{'switches':>10}{'switch_h':>10}{'max_blk':>9}{'mean_blk':>10}{'sweep_h':>9}"
    print(hdr)
    print("-" * len(hdr))
    for ws in wave_sizes:
        r = analyze(roster, ws, focus)
        print(
            f"{ws:>10}{r['switches']:>10}{r['switch_hours']:>9.1f}h"
            f"{r['max_block']:>9}{r['mean_block']:>10.1f}{r['sweep_hours']:>8.0f}h"
        )

    if args.wave_size:
        r = analyze(roster, args.wave_size, focus)
        print(f"\n{focus} block-length distribution (agents):")
        for size, count in r["block_hist"].items():
            print(f"  {count:>4} block(s) of {size:>4}")

    print(
        "\nNote: raising wave_size above the largest cohort forces k=1 everywhere,"
        "\ncollapses every key to 0.5, and degenerates the sort to alphabetical."
        "\nThe knob only goes down. Cohort ratios are invariant under wave_size —"
        "\nto change those, change the roster."
    )


if __name__ == "__main__":
    main()
