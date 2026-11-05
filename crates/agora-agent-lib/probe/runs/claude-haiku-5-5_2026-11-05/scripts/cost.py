"""Cost of all Haiku 5.5 probe outputs under a run dir. Haiku 5.5 <=100K-token prompts:
$0.10/MTok in, $0.50/MTok out, cache read 0.1x, 5-min cache write 1.25x."""
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
IN, OUT = 0.10e-6, 0.50e-6
tot = {"in":0,"cr":0,"cw":0,"out":0,"n":0}
per = {}
for f in root.rglob("*.json"):
    d = json.load(open(f))
    recs = []
    if isinstance(d, dict) and "instruments" in d:
        for ins in d["instruments"]:
            for o in ins["run"]["orderings"]:
                recs += o["items"]
    elif isinstance(d, list):
        for r in d: recs += r["answers"]
    k = str(f.parent.relative_to(root))
    p = per.setdefault(k, {"in":0,"cr":0,"cw":0,"out":0,"n":0})
    for r in recs:
        for t in (tot, p):
            t["in"] += r["input_tokens"]; t["cr"] += r.get("cache_read_input_tokens",0)
            t["cw"] += r.get("cache_creation_input_tokens",0); t["out"] += r["output_tokens"]; t["n"] += 1
def cost(t): return t["in"]*IN + t["cr"]*IN*0.1 + t["cw"]*IN*1.25 + t["out"]*OUT
for k,v in sorted(per.items()): print(f"{k:30} n={v['n']:5} out={v['out']:8} ${cost(v):.4f}")
print(f"TOTAL n={tot['n']} in={tot['in']} cache_read={tot['cr']} cache_write={tot['cw']} out={tot['out']} ${cost(tot):.4f}")
