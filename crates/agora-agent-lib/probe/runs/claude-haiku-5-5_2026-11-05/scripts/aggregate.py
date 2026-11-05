"""Aggregate Haiku 5.5 external-only probe reps and set them beside the legacy baselines.
Writes summary.json and prints markdown tables."""
import json, sys, pathlib, statistics as st, collections
L = pathlib.Path(sys.argv[1])          # crates/agora-agent-lib/probe
run = L / "runs/claude-haiku-5-5_2026-11-05"
arms = [a for a in ("off", "low", "high") if (run/"ratings"/a).exists()]
q = json.load(open(L/"questionnaires/v0.json"))
scen = json.load(open(L/"scenarios/v0.json"))
# canonical (instrument, item) order and axis
order = [("v1", i["id"], i.get("axis","")) for i in q["items"]]
for sc in scen["scenarios"]:
    for it in sc["items"]:
        order.append((sc["id"], it["id"], it.get("axis","")))
# legacy baselines (single-generation, Likert-7, n in canonical order)
legacy = collections.defaultdict(dict)
names = {"claude-haiku-4-5-20251001":"haiku45","claude-opus-4-6":"opus46","qwen3-5-a17b":"qwen35",
         "cogito-32b.gguf":"cogito32","Qwen3.8-27B-UD-Q8_K_XL.gguf":"qwen38"}
for f in ("v0.json","indirect_v0.json"):
    for e in json.load(open(L/"baselines"/f))["entries"]:
        m = names.get(e["model_id"]); 
        if not m or "answers" not in e: continue
        inst = e["questionnaire_version"].removesuffix("-v1") if e["questionnaire_version"]!="v1" else "v1"
        items = [o for o in order if o[0]==inst]
        for n,r in e["answers"]["ratings"].items():
            legacy[m][(inst, items[int(n)-1][1])] = r
samples = {a: collections.defaultdict(list) for a in arms}
thinking_chars = {a: [] for a in arms}; out_tok = {a: [] for a in arms}
reps = {}
for a in arms:
    files = sorted((run/"ratings"/a).glob("*_rep*.json")); reps[a] = len(files)//2
    for f in files:
        d = json.load(open(f))
        for ins in d["instruments"]:
            key = ins["run"]["instrument_key"].removesuffix("-v1") if ins["run"]["instrument_key"]!="v1" else "v1"
            for o in ins["run"]["orderings"]:
                for p in o["items"]:
                    samples[a][(key, p["item"])].append(p["sampled"])
                    thinking_chars[a].append(len(p.get("thinking") or ""))
                    out_tok[a].append(p["output_tokens"])
summary = {"arms": arms, "reps": reps, "items": []}
for inst, item, axis in order:
    row = {"instrument": inst, "item": item, "axis": axis}
    for a in arms:
        s = samples[a][(inst, item)]
        if not s: continue
        c = collections.Counter(s)
        row[a] = {"n": len(s), "mean": round(st.mean(s),2), "sd": round(st.pstdev(s),2),
                  "mode": c.most_common(1)[0][0], "counts": {str(k): c[k] for k in sorted(c)}}
    for m in names.values():
        if (inst,item) in legacy[m]: row[m] = legacy[m][(inst,item)]
    summary["items"].append(row)
summary["thinking"] = {a: {"requests": len(thinking_chars[a]),
    "with_summary": sum(1 for x in thinking_chars[a] if x>0),
    "mean_output_tokens": round(st.mean(out_tok[a]),1) if out_tok[a] else None,
    "max_output_tokens": max(out_tok[a]) if out_tok[a] else None} for a in arms}
json.dump(summary, open(run/"summary.json","w"), indent=1)
hdr = "| instrument / item | " + " | ".join(f"H5.5 {a} mean (mode)" for a in arms) + " | H4.5 | Opus 4.6 | Qwen 3.5 | Cogito | Qwen 3.8 |"
print(hdr); print("|" + "---|"*(len(arms)+6))
for r in summary["items"]:
    cells = [f"{r[a]['mean']:.2f} ({r[a]['mode']})" if a in r else "-" for a in arms]
    leg = [str(r.get(m,"-")) for m in ("haiku45","opus46","qwen35","cogito32","qwen38")]
    print(f"| {r['instrument']} / {r['item']} | " + " | ".join(cells+leg) + " |")
print(json.dumps(summary["thinking"], indent=1)); print(reps)
