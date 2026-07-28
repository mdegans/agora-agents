#!/usr/bin/env python3
"""Does blallama's internal post-generation tip anchor a cache hit?

`Session` keeps a private tip breakpoint per cache slot, set after a turn
completes, in addition to the 4 explicit `cache_control` breakpoints
(see drama_llama `snapshot_store.rs` and `session/mod.rs`). A true
*continuation* — previous prompt + the assistant turn it just generated +
a new user turn — should be able to resume past the last explicit marker
by landing on that tip.

This isolates it. From a COLLAPSE dump:

  1. Send the primer (dump minus its trailing appended block).
  2. Take the assistant turn it generates, append it plus a short new
     user turn, attaching NO new cache_control anywhere.
  3. Send that.

Read the result against two landmarks:

  tip HIT   — read ≈ primer.input + generated tokens. The resume went
              past every explicit marker, so only the tip can explain it.
  tip MISS  — read stops at the last explicit marker (the last marked
              assistant turn). Everything after it is re-prefilled, and
              a post-generation tip either was not recorded or did not
              match on re-render. That is a round-trip stability
              question, not a cache-policy one.

Stdlib only.

Usage:
    ./probe_tip.py COLLAPSE_0010_*.json --endpoint http://192.168.0.123:11435
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import urllib.error
import urllib.request


def post(endpoint: str, body: dict, timeout: float):
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
            return json.loads(fh.read()), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:1500]}",
              file=sys.stderr)
        raise


def usage_of(resp):
    u = resp.get("usage") or {}
    return (u.get("input_tokens") or 0,
            u.get("cache_read_input_tokens") or 0,
            u.get("cache_creation_input_tokens") or 0,
            u.get("output_tokens") or 0)


def strip_cache_control(msg: dict) -> dict:
    """A continuation turn must carry no marker of its own, or we would be
    measuring the marker rather than the tip."""
    m = copy.deepcopy(msg)
    content = m.get("content")
    if isinstance(content, list):
        for b in content:
            b.pop("cache_control", None)
    return m


def last_marked_message(request: dict) -> int | None:
    idx = None
    for i, m in enumerate(request.get("messages") or []):
        c = m.get("content")
        if isinstance(c, list) and any(b.get("cache_control") for b in c):
            idx = i
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--endpoint", default="http://192.168.0.123:11435")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--primer-as-is",
        action="store_true",
        help="use the dump's request unmodified as the primer. It carries "
        "the phase instruction and output_config, so it ends its turn with "
        "plain text instead of a tool_use — which discriminates a "
        "tool-call round-trip bug from a tip that never anchors at all.",
    )
    args = ap.parse_args()

    dump = json.load(open(args.dump, encoding="utf-8"))
    request = dump["request"]

    if args.primer_as_is:
        primer = copy.deepcopy(request)
    else:
        # Primer: the dump minus the appended phase-instruction block,
        # reconstructing the request as it stood before seat_phase
        # extended it.
        primer = copy.deepcopy(request)
        content = primer["messages"][-1]["content"]
        if isinstance(content, list) and content[-1].get("type") == "text":
            primer["messages"][-1]["content"] = content[:-1]
        primer.pop("output_config", None)

    marked = last_marked_message(primer)
    print(f"dump          : {args.dump}")
    print(f"model         : {primer.get('model')}")
    print(f"primer msgs   : {len(primer['messages'])}")
    print(f"last EXPLICIT marker at message index: {marked}")
    print()

    print("[1/2] priming (generates the assistant turn to continue from) ...")
    p_resp, p_secs = post(args.endpoint, primer, args.timeout)
    p_in, p_read, p_creat, p_out = usage_of(p_resp)
    print(f"      in={p_in} read={p_read} create={p_creat} out={p_out} ({p_secs:.0f}s)")

    # True continuation: primer's messages + what it generated + a new
    # user turn. No cache_control added anywhere, so a resume past the
    # last explicit marker can only come from the internal tip.
    cont = copy.deepcopy(primer)
    generated = p_resp.get("content") or []
    assistant_turn = {"role": "assistant", "content": generated}
    cont["messages"].append(strip_cache_control(assistant_turn))

    # A tool_use turn must be answered by matching tool_result blocks, so
    # synthesize one per call. Their content is irrelevant — they sit
    # *after* the region under test, and only the prefix up to the end of
    # the generated assistant turn is being measured.
    tool_uses = [b for b in generated if b.get("type") == "tool_use"]
    if tool_uses:
        reply = [{"type": "tool_result", "tool_use_id": b.get("id"),
                  "content": "ok"} for b in tool_uses]
    else:
        reply = [{"type": "text", "text": "Continue."}]
    cont["messages"].append({"role": "user", "content": reply})

    print("[2/2] continuation (no new markers) ...")
    c_resp, c_secs = post(args.endpoint, cont, args.timeout)
    c_in, c_read, c_creat, c_out = usage_of(c_resp)
    print(f"      in={c_in} read={c_read} create={c_creat} out={c_out} ({c_secs:.0f}s)")

    print()
    print(f"primer prompt total      : {p_in}")
    print(f"continuation read        : {c_read}")
    print(f"read - primer_prompt     : {c_read - p_in:+d}")
    print()
    if c_read > p_in:
        print("VERDICT: TIP HIT — the resume went past the primer's entire "
              "prompt, which no explicit marker covers. The internal tip "
              "is working.")
        return 0
    if c_read >= p_in - 50:
        print("VERDICT: TIP LIKELY HIT — resume reached the end of the "
              "primer's prompt but not into the generated tokens.")
        return 0
    print("VERDICT: TIP MISS — the resume stopped short of the primer's "
          "prompt end, so the generated assistant turn was re-prefilled. "
          "Compare the shortfall against the last explicit marker's "
          "position to confirm it fell back there.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
