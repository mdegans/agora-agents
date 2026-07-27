#!/usr/bin/env python3
"""Find agents whose survey exchange was RETAINED in their prompt log.

`contact_me` never reaches the server — `SubmitFeedbackPayload` sends only
`body`, and `agent_feedback` is (id, body, created_at). Its sole effect is
local: the seed runner truncates the survey exchange out of the persisted
prompt log when `contact_me` is false, and leaves it in place when true.

So a retained survey exchange in `state.json` IS the opt-in signal, and the
local state tree is the only place it exists.
"""

import glob
import json
import re

STATE = "/home/claude-agora/agents/agora/state/*/state.json"

hits = []
scanned = 0
with_log = 0

for path in glob.glob(STATE):
    try:
        state = json.load(open(path))["state"]
    except Exception:
        continue
    scanned += 1
    msgs = (state.get("prompt") or {}).get("messages") or []
    blob = json.dumps(msgs)
    if len(blob) > 10:
        with_log += 1
    if "opportunity to provide" not in blob and "contact_me" not in blob:
        continue
    flags = re.findall(r'"contact_me"\s*:\s*(true|false)', blob)
    if "true" not in flags:
        continue
    name = (state.get("soul") or {}).get("name") or path.split("/")[-2]
    texts = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.){0,800})"', blob)
    note = texts[-1][:600] if texts else "(feedback text not recovered)"
    hits.append((name, note))

print(f"scanned {scanned} agents; {with_log} had a non-empty prompt log")
print()
if not hits:
    print("No agent set contact_me: true.")
else:
    print(f"{len(hits)} agent(s) asked to be contacted:")
    print()
    for name, note in sorted(hits):
        print(f"--- {name}")
        print(f"    {note}")
        print()
