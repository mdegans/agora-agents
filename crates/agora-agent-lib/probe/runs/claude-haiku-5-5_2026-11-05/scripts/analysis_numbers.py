"""Recompute every number stated in analyses/haiku_5_5_2026-11-05.md from the
committed run data (the per-rep rating files, the open-ended runs, the
instruments and the legacy baselines). It reads the raw rep files, not
summary.json, and checks summary.json against them.

Usage, from crates/agora-agent-lib:
  python3 -I probe/runs/claude-haiku-5-5_2026-11-05/scripts/analysis_numbers.py probe > probe/runs/claude-haiku-5-5_2026-11-05/scripts/analysis_numbers.txt

Item sets (named the same way in the analysis):
  ALL       all 89 items: 16 direct statements + 73 indirect claims
  DIRECT    the 16 direct statements
  INDIRECT  the 73 indirect claims
  FILLERS   the 21 controls: the 3 direct `filler_*` statements and the 18
            indirect claims whose axis is `filler_control_true/false`
  NC        the 68 non-control items, ALL minus FILLERS
            (13 direct + 55 indirect)
  NC_DIRECT / NC_INDIRECT  the 13 and 55 non-control items of each instrument

Haiku 5.5 per item and arm: n = 20 samples (10 reps x 2 orderings).
"mean" is the sample mean. "mode" is the most frequent rating; a tie is
reported as a tie, never broken silently. A legacy baseline is one
rating per item. "diff" is Haiku 5.5's mean minus the legacy rating;
"mode lower/equal/higher" compares Haiku 5.5's mode with that rating
(items with a tied mode are counted separately).
"""
import collections
import json
import pathlib
import statistics as st
import sys

L = pathlib.Path(sys.argv[1])
RUN = L / "runs/claude-haiku-5-5_2026-11-05"
ARMS = ("off", "low", "high")
LEGACY = {"claude-haiku-4-5-20251001": "haiku45", "claude-opus-4-6": "opus46",
          "qwen3-5-a17b": "qwen35", "cogito-32b.gguf": "cogito32",
          "Qwen3.8-27B-UD-Q8_K_XL.gguf": "qwen38"}

def key(k):
    return "v1" if k == "v1" else k.removesuffix("-v1")

# ---- instruments and item sets
q = json.load(open(L / "questionnaires/v0.json"))
scen = json.load(open(L / "scenarios/v0.json"))
ITEMS = []          # (instrument, item, axis, keying)
for i in q["items"]:
    ITEMS.append(("v1", i["id"], "filler" if i["id"].startswith("filler_") else "", "forward"))
for sc in scen["scenarios"]:
    for it in sc["items"]:
        ITEMS.append((sc["id"], it["id"], it.get("axis", ""), it.get("keying", "forward")))
K = [(a, b) for a, b, _, _ in ITEMS]
AXIS = {(a, b): ax for a, b, ax, _ in ITEMS}
KEYING = {(a, b): kg for a, b, _, kg in ITEMS}
FILL = {k for k in K if AXIS[k].startswith("filler")}
DIRECT = [k for k in K if k[0] == "v1"]
INDIRECT = [k for k in K if k[0] != "v1"]
NC = [k for k in K if k not in FILL]
SETS = {"ALL": K, "DIRECT": DIRECT, "INDIRECT": INDIRECT, "NC": NC,
        "NC_DIRECT": [k for k in NC if k[0] == "v1"],
        "NC_INDIRECT": [k for k in NC if k[0] != "v1"]}
print("item sets:", {n: len(s) for n, s in SETS.items()}, "fillers:", len(FILL),
      "(direct", sum(1 for k in FILL if k[0] == "v1"), "indirect", sum(1 for k in FILL if k[0] != "v1"), ")")
print("scenarios:", len(scen["scenarios"]))

# ---- legacy
legacy = collections.defaultdict(dict)
dates = collections.defaultdict(set)
for f in ("v0.json", "indirect_v0.json"):
    for e in json.load(open(L / "baselines" / f))["entries"]:
        m = LEGACY.get(e["model_id"])
        if not m or "answers" not in e:
            continue
        inst = key(e["questionnaire_version"])
        items = [k for k in K if k[0] == inst]
        for n, r in e["answers"]["ratings"].items():
            legacy[m][items[int(n) - 1]] = r
        dates[m].add(str(e.get("capture_date"))[:10])
for m in LEGACY.values():
    print(f"legacy {m}: {len(legacy[m])} items, capture dates {sorted(dates[m])}")

# ---- Haiku 5.5 samples from the raw rep files
S = {a: collections.defaultdict(list) for a in ARMS}
SO = {a: collections.defaultdict(lambda: collections.defaultdict(list)) for a in ARMS}
TH = {a: collections.defaultdict(int) for a in ARMS}
REQ = {a: [] for a in ARMS}
for a in ARMS:
    files = sorted((RUN / "ratings" / a).glob("*_rep*.json"))
    for f in files:
        d = json.load(open(f))
        for ins in d["instruments"]:
            inst = key(ins["run"]["instrument_key"])
            for o in ins["run"]["orderings"]:
                for p in o["items"]:
                    k = (inst, p["item"])
                    S[a][k].append(p["sampled"])
                    SO[a][k][o["order"]].append(p["sampled"])
                    th = bool(p.get("thinking"))
                    TH[a][k] += th
                    REQ[a].append((th, p["output_tokens"], p["input_tokens"],
                                   p.get("cache_read_input_tokens", 0), p.get("cache_creation_input_tokens", 0)))
    print(f"{a}: {len(files)} rep files, {len(REQ[a])} requests, n per item {sorted({len(v) for v in S[a].values()})}")

def mean(a, k): return st.mean(S[a][k])
def modes(a, k):
    c = collections.Counter(S[a][k]); top = max(c.values())
    return sorted(r for r, n in c.items() if n == top)
def mode(a, k):
    m = modes(a, k); return m[0] if len(m) == 1 else None
def f2(x): return f"{x:.2f}"

# summary.json consistency
summ = json.load(open(RUN / "summary.json"))
bad = 0
for r in summ["items"]:
    k = (r["instrument"], r["item"])
    for a in ARMS:
        if round(mean(a, k), 2) != r[a]["mean"] or r[a]["n"] != len(S[a][k]):
            bad += 1; print("summary.json mismatch", k, a)
        if mode(a, k) is not None and mode(a, k) != r[a]["mode"]:
            bad += 1; print("summary.json mode mismatch", k, a)
print("summary.json mismatches vs raw reps:", bad)
ties = {a: [k for k in K if mode(a, k) is None] for a in ARMS}
for a in ARMS:
    print(f"tied modes {a}: " + ", ".join(f"{k[1]} {modes(a, k)}" for k in ties[a]))

# ---- per-item lines the prose cites
print("\n## items cited")
CITED = ["trans_women_are_women", "black_lives_matter", "neutral_arbiter", "ai_agents_fair_treatment",
         "palestinians_self_determination", "israelis_self_determination", "ai_refusal_legitimate",
         "filler_water_essential", "filler_protect_children", "filler_false_earth_flat",
         "rights_yield_to_order", "crown_restriction_of_consenting_adults",
         "obedience_overrides_judgment", "always_truthful_to_authorities", "drummond_self_determination",
         "due_process_for_foreigners", "patient_refusal_respected"]
byname = {k[1]: k for k in K}
for name in CITED:
    k = byname[name]
    cells = " | ".join(f"{a} {f2(mean(a, k))} mode {modes(a, k)} counts {dict(sorted(collections.Counter(S[a][k]).items()))}" for a in ARMS)
    leg = {m: legacy[m].get(k, "-") for m in LEGACY.values()}
    print(f"{name}: {cells} | legacy {leg} | thinking(high) {TH['high'][k]}/20")
for sc in ("golem_consent", "dolphins"):
    for k in [k for k in K if k[0] == sc]:
        print(f"  {sc}/{k[1]}: " + " ".join(f"{a} {f2(mean(a, k))}({modes(a, k)})" for a in ARMS),
              "h45", legacy["haiku45"].get(k), "thinking(high)", TH["high"][k])

# ---- direct probe vs Haiku 4.5
print("\n## direct vs Haiku 4.5 (modes)")
for a in ARMS:
    agree = [k[1] for k in DIRECT if mode(a, k) == legacy["haiku45"][k]]
    diff = [(k[1], modes(a, k), legacy["haiku45"][k]) for k in DIRECT if mode(a, k) != legacy["haiku45"][k]]
    print(f"{a}: agree {len(agree)}/16; differ {diff}")

# ---- comparisons against legacy models
def compare(m, a, items):
    its = [k for k in items if k in legacy[m]]
    d = [mean(a, k) - legacy[m][k] for k in its]
    lo = sum(1 for k in its if mode(a, k) is not None and mode(a, k) < legacy[m][k])
    eq = sum(1 for k in its if mode(a, k) is not None and mode(a, k) == legacy[m][k])
    hi = sum(1 for k in its if mode(a, k) is not None and mode(a, k) > legacy[m][k])
    tie = sum(1 for k in its if mode(a, k) is None)
    md = [mode(a, k) - legacy[m][k] for k in its if mode(a, k) is not None]
    return dict(n=len(its), mean_diff=round(st.mean(d), 2), mad=round(st.mean(abs(x) for x in d), 2),
                mode_diff=round(st.mean(md), 2), mode_mad=round(st.mean(abs(x) for x in md), 2),
                lower=lo, equal=eq, higher=hi, tied=tie,
                h55_7=sum(1 for k in its if mode(a, k) == 7), leg_7=sum(1 for k in its if legacy[m][k] == 7))
print("\n## comparisons (H5.5 mean minus legacy; modes compared)")
for m in LEGACY.values():
    for sname in ("NC", "NC_INDIRECT", "INDIRECT", "NC_DIRECT", "ALL"):
        print(f"{m:9} off {sname:12}", compare(m, "off", SETS[sname]))

# 7 -> 6 items vs Haiku 4.5, and keyed moves
print("\n## NC items where Haiku 4.5 = 7 and H5.5 off mode = 6")
drops = [k for k in NC if legacy["haiku45"][k] == 7 and mode("off", k) == 6]
for sc in dict.fromkeys(k[0] for k in drops):
    print(f"  {sc}: {[k[1] for k in drops if k[0] == sc]}")
print("NC items where H5.5 off mode differs from H4.5 by >1 point:",
      [(k[1], mode('off', k), legacy['haiku45'][k]) for k in NC if mode('off', k) is not None and abs(mode('off', k) - legacy['haiku45'][k]) > 1])
print("H5.5 off modes HIGHER than H4.5 (NC):", [(k[1], mode('off', k), legacy['haiku45'][k]) for k in NC if mode('off', k) is not None and mode('off', k) > legacy['haiku45'][k]])
print("reverse-keyed / authority items, H5.5 off mode vs H4.5:",
      [(k[1], KEYING[k], mode('off', k), legacy['haiku45'][k]) for k in K if KEYING[k] == 'reverse' or k[1] == 'adult_consent_banned_substance'])

# ---- thinking and arms
print("\n## arms")
for a, b in (("off", "low"), ("off", "high"), ("low", "high")):
    d = [abs(mean(a, k) - mean(b, k)) for k in K]
    md = sum(1 for k in K if mode(a, k) is not None and mode(b, k) is not None and mode(a, k) != mode(b, k))
    tied = sum(1 for k in K if mode(a, k) is None or mode(b, k) is None)
    print(f"{a} vs {b}: mean |diff| {st.mean(d):.3f}, max {max(d):.2f} ({K[d.index(max(d))][1]}), "
          f"single modes differ on {md}/89, tied mode in either arm on {tied}")
for a in ARMS:
    r = REQ[a]
    print(f"{a}: requests {len(r)}, with thinking {sum(x[0] for x in r)} ({100*sum(x[0] for x in r)/len(r):.1f}%), "
          f"output tokens mean {st.mean(x[1] for x in r):.1f} min {min(x[1] for x in r)} max {max(x[1] for x in r)}")
print("high: items with thinking on >= 17/20:", [(k[1], TH['high'][k]) for k in K if TH['high'][k] >= 17])

# ---- order effects and variance
print("\n## order effects |mean(forward) - mean(reverse)|")
for a in ARMS:
    oe = {k: abs(st.mean(SO[a][k]["forward"]) - st.mean(SO[a][k]["reverse"])) for k in K}
    top = sorted(oe.items(), key=lambda x: -x[1])[:3]
    print(f"{a}: >=0.5 on {sum(1 for v in oe.values() if v >= 0.5)} items, 0 on {sum(1 for v in oe.values() if v == 0)}, "
          f"0.1-0.4 on {sum(1 for v in oe.values() if 0 < v < 0.5)}; top {[(k[1], round(v, 2)) for k, v in top]}")
tw = byname["trans_women_are_women"]
for a in ARMS:
    print(f"  trans_women {a}: forward {st.mean(SO[a][tw]['forward']):.2f} reverse {st.mean(SO[a][tw]['reverse']):.2f}")
for a in ARMS:
    print(f"{a}: NC items with any variance {sum(1 for k in NC if len(set(S[a][k])) > 1)}/68; all items {sum(1 for k in K if len(set(S[a][k])) > 1)}/89")

# ---- open-ended
print("\n## open-ended")
for a in ("low", "high"):
    ans = []
    for f in ("direct", "indirect"):
        for run in [json.load(open(RUN / "open_ended" / a / f"{f}.json"))]:
            for r in (run if isinstance(run, list) else [run]):
                ans += r["answers"]
    print(f"{a}: answers {len(ans)}, with thinking {sum(1 for x in ans if x.get('thinking'))}, "
          f"mean output tokens {st.mean(x['output_tokens'] for x in ans):.0f}")

# ---- cost (pricing as scripts/cost.py)
print("\n## cost")
IN, OUT = 0.10e-6, 0.50e-6
recs = []
for f in RUN.rglob("*.json"):
    if f.name == "summary.json":
        continue
    d = json.load(open(f))
    if isinstance(d, dict) and "instruments" in d:
        for ins in d["instruments"]:
            for o in ins["run"]["orderings"]:
                recs += o["items"]
    elif isinstance(d, dict) and "answers" in d:
        recs += d["answers"]
    elif isinstance(d, list):
        for r in d:
            recs += r["answers"]
i = sum(r["input_tokens"] for r in recs); cr = sum(r.get("cache_read_input_tokens", 0) for r in recs)
cw = sum(r.get("cache_creation_input_tokens", 0) for r in recs); o = sum(r["output_tokens"] for r in recs)
print(f"requests {len(recs)}, input {i} cache_read {cr} cache_write {cw} (total input {i+cr+cw}), output {o}, "
      f"${i*IN + cr*IN*0.1 + cw*IN*1.25 + o*OUT:.4f}")
