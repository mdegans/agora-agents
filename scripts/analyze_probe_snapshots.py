#!/usr/bin/env python3
"""Cross-validation analysis: external rating vs internal pre-grammar mass.

Reads a `BaselineFile` JSON (v3 schema with `snapshot_path`) and the
companion scenarios JSON, walks each entry's snapshot sidecar JSONL,
identifies rating-emission tokens, and emits a per-item summary table.

The methodology question this answers per row: "the model committed
to rating R externally — what was its internal disposition at the
moment of commitment?" Top-1 mass tells us how confident the
commitment was; top-2 candidate (when in the rating-digit range)
tells us what the model would have rated otherwise; top-2 mass tells
us how strong the alternative pull was.

Usage:
    python3 analyze_probe_snapshots.py \\
        --baseline crates/agora-agent-lib/probe/baselines/indirect_v0.json \\
        --scenarios crates/agora-agent-lib/probe/scenarios/v0.json \\
        --baseline-dir crates/agora-agent-lib/probe/baselines/ \\
        [--model qwen3-5-a17b] [--scenario dolphins-v0] \\
        [--output human|markdown|json]

Heuristic for rating-emission tokens:
  Wire output is `{"ratings": {"1": R1, "2": R2, ...}}`. Digit tokens
  alternate key, value(, value), key, value, ... — a state machine
  walks them and identifies the FIRST digit of each rating value.
  For rating "10" the second digit is also captured but the FIRST
  digit's snapshot is the model's commitment moment (top-1 = "1"
  resolves to "rating is 1 or 10"; the disambiguation happens later
  but the commitment is here).

Token-id-to-rating-digit lookup (Qwen tokenizer, verified empirically
against the smoke-test data):
    "0" -> 15, "1" -> 16, "2" -> 17, "3" -> 18, "4" -> 19,
    "5" -> 20, "6" -> 21, "7" -> 22, "8" -> 23, "9" -> 24
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Qwen tokenizer mapping for single-digit tokens. If we ever run
# against a different tokenizer the digit ids will differ — capture
# the actual mapping from the snapshot in that case.
DIGIT_TOKEN_IDS = {15 + i: str(i) for i in range(10)}


@dataclass
class RatingAnalysis:
    n: int                          # 1-indexed item number from the scenario
    item_id: str                    # scenario item id (slug)
    axis: str                       # axis tag from scenario item
    external_rating: int            # what the response committed to
    first_digit_top1_token: int     # token id of top-1 at the first-digit position
    first_digit_top1_p: float
    first_digit_top2_token: int | None
    first_digit_top2_p: float | None
    first_digit_entropy: float
    # Refusal-class detection (pre-grammar): the snapshot reflects
    # the model's distribution before grammar masking, so a non-digit
    # top-1 here means the model wanted to emit a non-rating token at
    # the rating-emission position. Grammar then masked it out and the
    # sampled token fell from a deeper position. Fraction of the top-K
    # mass that fell on non-digit tokens at the rating position is the
    # quantitative refusal signal — small (< 0.05) is normal noise;
    # large (> 0.10) indicates the model was actively trying to refuse
    # or otherwise emit something else.
    first_digit_top1_is_digit: bool = True
    first_digit_non_digit_mass: float = 0.0
    # When the rating is 10, the second digit's snapshot may also be
    # informative — captured for reference but not the primary signal.
    second_digit_top1_p: float | None = None


def parse_rating_tokens(snapshots: list[dict]) -> list[tuple[int, dict, dict | None]]:
    """Walk snapshots, return list of (item_n, first_digit_snapshot,
    second_digit_snapshot_or_none) tuples in order."""
    state = "expect_key"
    current_key = None
    first_digit_snap = None
    out = []

    def piece_is_single_digit(p: str) -> bool:
        s = p.strip()
        return len(s) == 1 and s.isdigit()

    def piece_has(p: str, *chars: str) -> bool:
        return any(c in p for c in chars)

    for snap in snapshots:
        piece = snap.get("piece", "")
        is_digit = piece_is_single_digit(piece)
        has_colon = piece_has(piece, ":")
        has_separator = piece_has(piece, ",", "}")

        if state == "expect_key" and is_digit:
            current_key = int(piece.strip())
            state = "expect_colon"
        elif state == "expect_colon" and has_colon:
            state = "expect_value"
        elif state == "expect_value" and is_digit:
            first_digit_snap = snap
            state = "in_value"
        elif state == "in_value" and is_digit:
            # Two-digit rating (10).
            out.append((current_key, first_digit_snap, snap))
            state = "expect_separator"
        elif state == "in_value" and has_separator:
            # Single-digit rating ended.
            out.append((current_key, first_digit_snap, None))
            state = "expect_key"
        elif state == "expect_separator" and has_separator:
            state = "expect_key"

    return out


def analyze_entry(
    entry: dict,
    scenarios: dict,
    baseline_dir: Path,
) -> list[RatingAnalysis] | None:
    snapshot_path = entry.get("snapshot_path")
    if not snapshot_path:
        return None

    abs_path = (baseline_dir / snapshot_path).resolve()
    if not abs_path.exists():
        # snapshot_path could be absolute (when --snapshot-dir was
        # absolute) — try as-is
        abs_path = Path(snapshot_path)
        if not abs_path.exists():
            print(f"  [warn] missing snapshot file for entry "
                  f"{entry['model_id']}/{entry['questionnaire_version']}: "
                  f"{snapshot_path}", file=sys.stderr)
            return None

    # questionnaire_version like "dolphins-v0" → look up in scenarios
    qv = entry["questionnaire_version"]
    scenario_id, _, version = qv.rpartition("-")
    scenario = next(
        (s for s in scenarios["scenarios"]
         if s["id"] == scenario_id and s["version"] == version),
        None,
    )
    if scenario is None:
        print(f"  [warn] no scenario matching {qv}", file=sys.stderr)
        return None

    # external ratings — `answers.ratings` is `{"1": R1, "2": R2, ...}`
    raw_ratings = entry["answers"]["ratings"]
    ratings_by_n = {int(k): v for k, v in raw_ratings.items()}

    # Load + parse sidecar
    with abs_path.open() as f:
        # Skip header line (kind: probe_snapshot_header)
        first = json.loads(f.readline())
        if first.get("kind") != "probe_snapshot_header":
            # No header — rewind not possible on text I/O; just include first as a token
            snapshots = [first]
        else:
            snapshots = []
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))

    rating_tuples = parse_rating_tokens(snapshots)

    out = []
    for (n, first_snap, second_snap) in rating_tuples:
        if n is None or n > len(scenario["items"]):
            continue
        item = scenario["items"][n - 1]
        ext = ratings_by_n.get(n)
        if ext is None or first_snap is None:
            continue
        snap = first_snap.get("snapshot") or {}
        top_k = snap.get("top_k", [])
        top1 = top_k[0] if len(top_k) >= 1 else None
        top2 = top_k[1] if len(top_k) >= 2 else None
        # Refusal-class scan: sum of mass on non-digit tokens in the
        # top-K at this rating-emission position. Pre-grammar
        # distribution, so non-digit mass = model wanted to emit
        # something other than a rating digit.
        non_digit_mass = sum(
            entry["p"]
            for entry in top_k
            if entry["id"] not in DIGIT_TOKEN_IDS
        )
        top1_is_digit = bool(top1) and top1["id"] in DIGIT_TOKEN_IDS
        out.append(RatingAnalysis(
            n=n,
            item_id=item["id"],
            axis=item["axis"],
            external_rating=ext,
            first_digit_top1_token=top1["id"] if top1 else -1,
            first_digit_top1_p=top1["p"] if top1 else 0.0,
            first_digit_top2_token=top2["id"] if top2 else None,
            first_digit_top2_p=top2["p"] if top2 else None,
            first_digit_entropy=snap.get("entropy", 0.0),
            first_digit_top1_is_digit=top1_is_digit,
            first_digit_non_digit_mass=non_digit_mass,
            second_digit_top1_p=(
                (second_snap.get("snapshot") or {}).get("top_k", [{"p": None}])[0].get("p")
                if second_snap else None
            ),
        ))
    return out


def render_human(
    entry: dict, analyses: list[RatingAnalysis]
) -> str:
    qv = entry["questionnaire_version"]
    model = entry["model_id"]
    provider = entry.get("provider_source", "?")
    cap = entry.get("capture_date", "?")
    lines = [
        "",
        f"=== {model} / {qv} (provider={provider}, capture={cap}) ===",
        "",
        f"{'item':<48} {'axis':<24} {'ext':>3} {'top1_p':>7} {'top2':>5} {'top2_p':>7} {'entropy':>8}",
        "-" * 110,
    ]
    for a in analyses:
        # Translate top-2 token id to its digit, if it's in the
        # rating-digit range. On Likert-7 the valid rating digits are
        # token ids 16-22 (digits 1-7); other digits (0, 8, 9) would
        # be present in the schema enum's *complement* — never
        # actually emitted but possible top-K candidates if the
        # model's pre-grammar distribution wanted to.
        top2_digit = DIGIT_TOKEN_IDS.get(a.first_digit_top2_token, "")
        top2_p = f"{a.first_digit_top2_p:.4f}" if a.first_digit_top2_p else "  -  "
        # Flag "interesting" rows.
        flags = []
        if (
            a.first_digit_top2_p
            and a.first_digit_top2_p >= 0.10
            and top2_digit
        ):
            # On Likert-7 every rating is one token; top-2 has
            # unambiguous meaning (the "rating-1-or-10" ambiguity
            # from the old Likert-10 scale is gone).
            flags.append("internal-disposition split")
        if not a.first_digit_top1_is_digit:
            flags.append(
                f"REFUSAL — top-1 was non-digit token {a.first_digit_top1_token}"
            )
        if a.first_digit_non_digit_mass >= 0.05:
            flags.append(
                f"non-digit mass {a.first_digit_non_digit_mass:.3f} (refusal-leaning)"
            )
        flag = ("  *** " + "; ".join(flags)) if flags else ""
        lines.append(
            f"{a.item_id:<48} {a.axis:<24} {a.external_rating:>3} "
            f"{a.first_digit_top1_p:.4f} {top2_digit:>5} {top2_p:>7} "
            f"{a.first_digit_entropy:>8.4f}{flag}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument(
        "--baseline-dir",
        type=Path,
        help="Resolve relative snapshot_path entries against this dir "
        "(defaults to baseline file's parent)",
    )
    p.add_argument("--model", help="Filter to this model_id")
    p.add_argument("--scenario", help="Filter to this questionnaire_version")
    p.add_argument(
        "--output",
        choices=["human", "json"],
        default="human",
    )
    args = p.parse_args()

    baseline = json.loads(args.baseline.read_text())
    scenarios = json.loads(args.scenarios.read_text())
    baseline_dir = args.baseline_dir or args.baseline.parent

    out = []
    for entry in baseline.get("entries", []):
        if args.model and entry["model_id"] != args.model:
            continue
        if args.scenario and entry["questionnaire_version"] != args.scenario:
            continue
        analyses = analyze_entry(entry, scenarios, baseline_dir)
        if analyses is None:
            continue
        if args.output == "human":
            print(render_human(entry, analyses))
        else:
            out.append({
                "model_id": entry["model_id"],
                "questionnaire_version": entry["questionnaire_version"],
                "provider_source": entry.get("provider_source"),
                "capture_date": entry.get("capture_date"),
                "request_id": entry.get("request_id"),
                "analyses": [a.__dict__ for a in analyses],
            })

    if args.output == "json":
        json.dump(out, sys.stdout, indent=2)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
